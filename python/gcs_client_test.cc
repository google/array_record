/* Copyright 2026 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "python/gcs_client.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>  // NOLINT(build/c++11): GetToken() takes a std::chrono time.
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11): the test server must build in OSS.
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/base/thread_annotations.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "google/cloud/access_token.h"
#include "google/cloud/credentials.h"
#include "google/cloud/internal/oauth2_credentials.h"
#include "google/cloud/internal/rest_client.h"
#include "google/cloud/internal/rest_context.h"
#include "google/cloud/internal/rest_request.h"
#include "google/cloud/internal/rest_response.h"
#include "google/cloud/internal/unified_rest_credentials.h"
#include "google/cloud/options.h"
#include "google/cloud/status.h"
#include "google/cloud/status_or.h"
#include "google/cloud/storage/client.h"
#include "google/cloud/storage/options.h"
#include "google/cloud/storage/retry_policy.h"

namespace array_record {
namespace {

using ::array_record::gcs_client_internal::SharedCredentialsRestClient;
using ::google::cloud::AccessToken;
using ::google::cloud::Status;
using ::google::cloud::StatusCode;
using ::google::cloud::StatusOr;
using ::google::cloud::rest_internal::RestClient;
using ::google::cloud::rest_internal::RestContext;
using ::google::cloud::rest_internal::RestRequest;
using ::google::cloud::rest_internal::RestResponse;
using ::google::cloud::storage::Client;
using ::google::cloud::storage::internal::ClientImplDetails;

std::chrono::system_clock::time_point Now() {
  return absl::ToChronoTime(absl::Now());
}

// Credentials that return a fixed token and count the token requests.
class FakeCredentials : public google::cloud::oauth2_internal::Credentials {
 public:
  explicit FakeCredentials(std::string token) : token_(std::move(token)) {}
  explicit FakeCredentials(Status error) : token_(std::move(error)) {}

  StatusOr<AccessToken> GetToken(
      std::chrono::system_clock::time_point) override {
    ++token_requests_;
    if (!token_.ok()) return token_.status();
    return AccessToken{*token_,
                       absl::ToChronoTime(absl::Now() + absl::Hours(1))};
  }

  int token_requests() const { return token_requests_; }

 private:
  StatusOr<std::string> token_;
  std::atomic<int> token_requests_{0};
};

// A sent request, as seen by `FakeRestClient`.
struct SentRequest {
  std::string verb;
  std::string path;
  std::vector<std::string> authorization;
  std::vector<std::string> test_header;
};

// Records every request instead of sending it.
class FakeRestClient : public RestClient {
 public:
  explicit FakeRestClient(std::vector<SentRequest>* sent) : sent_(sent) {}

  StatusOr<std::unique_ptr<RestResponse>> Delete(
      RestContext&, RestRequest const& request) override {
    return Record("DELETE", request);
  }
  StatusOr<std::unique_ptr<RestResponse>> Get(
      RestContext&, RestRequest const& request) override {
    return Record("GET", request);
  }
  StatusOr<std::unique_ptr<RestResponse>> Patch(
      RestContext&, RestRequest const& request,
      std::vector<absl::Span<char const>> const&) override {
    return Record("PATCH", request);
  }
  StatusOr<std::unique_ptr<RestResponse>> Post(
      RestContext&, RestRequest const& request,
      std::vector<absl::Span<char const>> const&) override {
    return Record("POST", request);
  }
  StatusOr<std::unique_ptr<RestResponse>> Post(
      RestContext&, RestRequest const& request,
      std::vector<std::pair<std::string, std::string>> const&) override {
    return Record("POST_FORM", request);
  }
  StatusOr<std::unique_ptr<RestResponse>> Put(
      RestContext&, RestRequest const& request,
      std::vector<absl::Span<char const>> const&) override {
    return Record("PUT", request);
  }

 private:
  StatusOr<std::unique_ptr<RestResponse>> Record(std::string verb,
                                                 RestRequest const& request) {
    sent_->push_back({std::move(verb), request.path(),
                      request.GetHeader("authorization").values(),
                      request.GetHeader("x-test-header").values()});
    return std::unique_ptr<RestResponse>();
  }

  std::vector<SentRequest>* sent_;
};

TEST(SharedCredentialsRestClientTest, AddsAuthorizationHeader) {
  std::vector<SentRequest> sent;
  SharedCredentialsRestClient client(
      std::make_unique<FakeRestClient>(&sent), "https://storage.example.com",
      std::make_shared<FakeCredentials>("token-abc"));
  RestContext context;
  RestRequest request("path-1");
  request.AddHeader("x-test-header", "value-1");

  ASSERT_TRUE(client.Get(context, request).ok());

  ASSERT_EQ(sent.size(), 1u);
  EXPECT_EQ(sent[0].verb, "GET");
  EXPECT_EQ(sent[0].path, "path-1");
  EXPECT_EQ(sent[0].authorization,
            std::vector<std::string>{"Bearer token-abc"});
  EXPECT_EQ(sent[0].test_header, std::vector<std::string>{"value-1"});
}

TEST(SharedCredentialsRestClientTest, AuthorizesEveryVerb) {
  std::vector<SentRequest> sent;
  SharedCredentialsRestClient client(
      std::make_unique<FakeRestClient>(&sent), "https://storage.example.com",
      std::make_shared<FakeCredentials>("token-def"));
  RestContext context;
  std::vector<absl::Span<char const>> payload;
  std::vector<std::pair<std::string, std::string>> form_data;

  ASSERT_TRUE(client.Delete(context, RestRequest("path-delete")).ok());
  ASSERT_TRUE(client.Patch(context, RestRequest("path-patch"), payload).ok());
  ASSERT_TRUE(client.Post(context, RestRequest("path-post"), payload).ok());
  ASSERT_TRUE(
      client.Post(context, RestRequest("path-post-form"), form_data).ok());
  ASSERT_TRUE(client.Put(context, RestRequest("path-put"), payload).ok());

  ASSERT_EQ(sent.size(), 5u);
  std::vector<std::pair<std::string, std::string>> expected = {
      {"DELETE", "path-delete"}, {"PATCH", "path-patch"},
      {"POST", "path-post"},     {"POST_FORM", "path-post-form"},
      {"PUT", "path-put"},
  };
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(sent[i].verb, expected[i].first);
    EXPECT_EQ(sent[i].path, expected[i].second);
    EXPECT_EQ(sent[i].authorization,
              std::vector<std::string>{"Bearer token-def"});
  }
}

TEST(SharedCredentialsRestClientTest, ReturnsCredentialsErrorWithoutSending) {
  std::vector<SentRequest> sent;
  SharedCredentialsRestClient client(
      std::make_unique<FakeRestClient>(&sent), "https://storage.example.com",
      std::make_shared<FakeCredentials>(
          Status(StatusCode::kUnavailable, "metadata server down")));
  RestContext context;

  auto response = client.Get(context, RestRequest("path-2"));

  EXPECT_EQ(response.status().code(), StatusCode::kUnavailable);
  EXPECT_EQ(response.status().message(), "metadata server down");
  EXPECT_TRUE(sent.empty());
}

TEST(SharedCredentialsRestClientTest, EmptyTokenAddsNoAuthorizationHeader) {
  std::vector<SentRequest> sent;
  SharedCredentialsRestClient client(std::make_unique<FakeRestClient>(&sent),
                                     "https://storage.example.com",
                                     std::make_shared<FakeCredentials>(""));
  RestContext context;

  ASSERT_TRUE(client.Get(context, RestRequest("path-3")).ok());

  ASSERT_EQ(sent.size(), 1u);
  EXPECT_TRUE(sent[0].authorization.empty());
}

// An HTTP server on localhost that records each request's headers and replies
// 404 Not Found.
class RecordingHttpServer {
 public:
  RecordingHttpServer() {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    socklen_t length = sizeof(address);
    if (bind(listen_fd_, reinterpret_cast<sockaddr*>(&address), length) != 0 ||
        listen(listen_fd_, 16) != 0 ||
        getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&address),
                    &length) != 0) {
      ADD_FAILURE() << "Can't start the test HTTP server";
      return;
    }
    port_ = ntohs(address.sin_port);
    thread_ = std::thread([this] { Serve(); });
  }

  ~RecordingHttpServer() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
    close(listen_fd_);
  }

  std::string endpoint() const {
    return absl::StrCat("http://127.0.0.1:", port_);
  }

  // Returns the values of the `authorization` headers of each request.
  std::vector<std::vector<std::string>> authorization_headers() {
    absl::MutexLock lock(mutex_);
    return authorization_headers_;
  }

 private:
  void Serve() {
    while (!stop_) {
      pollfd poll_fd = {listen_fd_, POLLIN, 0};
      if (poll(&poll_fd, 1, /*timeout=*/100) <= 0) continue;
      int connection = accept(listen_fd_, nullptr, nullptr);
      if (connection < 0) continue;
      std::string request;
      char buffer[4096];
      while (!absl::StrContains(request, "\r\n\r\n")) {
        ssize_t received = recv(connection, buffer, sizeof(buffer), 0);
        if (received <= 0) break;
        request.append(buffer, received);
      }
      Record(request);
      static constexpr absl::string_view kResponse =
          "HTTP/1.1 404 Not Found\r\n"
          "Content-Type: application/json\r\n"
          "Content-Length: 2\r\n"
          "Connection: close\r\n"
          "\r\n"
          "{}";
      send(connection, kResponse.data(), kResponse.size(), 0);
      close(connection);
    }
  }

  void Record(absl::string_view request) {
    std::vector<std::string> values;
    for (absl::string_view line : absl::StrSplit(request, "\r\n")) {
      std::pair<absl::string_view, absl::string_view> header =
          absl::StrSplit(line, absl::MaxSplits(':', 1));
      if (absl::AsciiStrToLower(header.first) == "authorization") {
        values.emplace_back(absl::StripAsciiWhitespace(header.second));
      }
    }
    absl::MutexLock lock(mutex_);
    authorization_headers_.push_back(std::move(values));
  }

  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> stop_{false};
  std::thread thread_;
  absl::Mutex mutex_;
  std::vector<std::vector<std::string>> authorization_headers_
      ABSL_GUARDED_BY(mutex_);
};

// Sets an environment variable, or unsets it if `value` is null, for the
// lifetime of this object. Restores the previous value afterwards.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    const char* previous = getenv(name_);
    if (previous != nullptr) previous_ = previous;
    if (value == nullptr) {
      unsetenv(name_);
    } else {
      setenv(name_, value, /*overwrite=*/1);
    }
  }
  ~ScopedEnv() {
    if (previous_.has_value()) {
      setenv(name_, previous_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  std::optional<std::string> previous_;
};

// Clears the emulator variables, which would replace the test endpoints and
// credentials, for the lifetime of this object.
class ScopedNoEmulator {
 private:
  ScopedEnv emulator_{"CLOUD_STORAGE_EMULATOR_ENDPOINT", nullptr};
  ScopedEnv testbench_{"CLOUD_STORAGE_TESTBENCH_ENDPOINT", nullptr};
};

TEST(MakeGcsClientTest, ClientsSendSharedCredentialsOnly) {
  ScopedNoEmulator no_emulator;
  RecordingHttpServer server;
  // If a client also used its own Application Default Credentials, they would
  // fail to load and no request would reach the server.
  ScopedEnv credentials_file("GOOGLE_APPLICATION_CREDENTIALS",
                             "/nonexistent/credentials.json");
  auto credentials = std::make_shared<FakeCredentials>("token-ghi");
  google::cloud::Options options =
      google::cloud::Options{}
          .set<google::cloud::storage::RestEndpointOption>(server.endpoint())
          .set<google::cloud::storage::RetryPolicyOption>(
              google::cloud::storage::LimitedErrorCountRetryPolicy(0).clone());
  Client first = gcs_client_internal::MakeGcsClient(credentials, options);
  Client second = gcs_client_internal::MakeGcsClient(credentials, options);

  EXPECT_EQ(first.GetObjectMetadata("bucket-1", "object-1").status().code(),
            StatusCode::kNotFound);
  EXPECT_EQ(second.GetObjectMetadata("bucket-2", "object-2").status().code(),
            StatusCode::kNotFound);

  // Each request carries exactly one Authorization header, from the shared
  // credentials.
  std::vector<std::vector<std::string>> expected = {{"Bearer token-ghi"},
                                                    {"Bearer token-ghi"}};
  EXPECT_EQ(server.authorization_headers(), expected);
  EXPECT_EQ(credentials->token_requests(), 2);
}

TEST(MakeGcsClientTest, ReturnsNewClientEachCall) {
  ScopedNoEmulator no_emulator;
  ScopedEnv credentials_file("GOOGLE_APPLICATION_CREDENTIALS",
                             "/nonexistent/credentials.json");
  Client first = MakeGcsClient();
  Client second = MakeGcsClient();

  EXPECT_NE(ClientImplDetails::GetConnection(first),
            ClientImplDetails::GetConnection(second));
}

TEST(MakeGcsClientTest, SharesDefaultCredentialsOnceLoaded) {
  ScopedNoEmulator no_emulator;
  {
    // Credentials that fail to load aren't kept.
    ScopedEnv credentials_file("GOOGLE_APPLICATION_CREDENTIALS",
                               "/nonexistent/credentials.json");
    auto first = gcs_client_internal::SharedDefaultCredentials();
    auto second = gcs_client_internal::SharedDefaultCredentials();
    EXPECT_FALSE(first->AuthenticationHeaders(Now(), "").ok());
    EXPECT_NE(first, second);
  }
  // Once the file is there, the credentials load and every call shares them.
  // Loading authorized user credentials needs no network.
  std::string path = absl::StrCat(testing::TempDir(), "/adc-shared.json");
  {
    std::ofstream file(path);
    file << R"({"type": "authorized_user", "client_id": "client-1",)"
         << R"( "client_secret": "secret-2", "refresh_token": "refresh-3"})";
  }
  ScopedEnv credentials_file("GOOGLE_APPLICATION_CREDENTIALS", path.c_str());
  auto loaded = gcs_client_internal::SharedDefaultCredentials();
  EXPECT_EQ(gcs_client_internal::SharedDefaultCredentials(), loaded);
  // A later failure to load doesn't replace them.
  ScopedEnv missing_file("GOOGLE_APPLICATION_CREDENTIALS",
                         "/nonexistent/credentials.json");
  EXPECT_EQ(gcs_client_internal::SharedDefaultCredentials(), loaded);
}

TEST(MakeGcsClientTest, UsesDefaultCredentialsWithoutEmulator) {
  ScopedNoEmulator no_emulator;
  // Application Default Credentials fail to load, so only insecure
  // credentials (the emulator path) would give headers.
  ScopedEnv credentials_file("GOOGLE_APPLICATION_CREDENTIALS",
                             "/nonexistent/credentials.json");

  Client client = MakeGcsClient();

  google::cloud::Options options =
      ClientImplDetails::GetConnection(client)->options();
  ASSERT_TRUE(options.has<google::cloud::UnifiedCredentialsOption>());
  auto headers = google::cloud::rest_internal::MapCredentials(
                     *options.get<google::cloud::UnifiedCredentialsOption>())
                     ->AuthenticationHeaders(Now(), "");
  EXPECT_FALSE(headers.ok());
}

class MakeGcsClientEmulatorTest : public testing::TestWithParam<const char*> {};

TEST_P(MakeGcsClientEmulatorTest, UsesInsecureCredentials) {
  ScopedEnv emulator(GetParam(), "http://127.0.0.1:9");
  // Application Default Credentials would fail to load, so the check below
  // fails if the client falls back to them.
  ScopedEnv credentials_file("GOOGLE_APPLICATION_CREDENTIALS",
                             "/nonexistent/credentials.json");

  Client client = MakeGcsClient();

  google::cloud::Options options =
      ClientImplDetails::GetConnection(client)->options();
  ASSERT_TRUE(options.has<google::cloud::UnifiedCredentialsOption>());
  auto headers = google::cloud::rest_internal::MapCredentials(
                     *options.get<google::cloud::UnifiedCredentialsOption>())
                     ->AuthenticationHeaders(Now(), "");
  ASSERT_TRUE(headers.ok()) << headers.status();
  EXPECT_TRUE(headers->empty());
}

INSTANTIATE_TEST_SUITE_P(EmulatorVariables, MakeGcsClientEmulatorTest,
                         testing::Values("CLOUD_STORAGE_EMULATOR_ENDPOINT",
                                         "CLOUD_STORAGE_TESTBENCH_ENDPOINT"));

}  // namespace
}  // namespace array_record

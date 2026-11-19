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

#include <chrono>  // NOLINT(build/c++11): GetToken() takes a std::chrono time.
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "google/cloud/access_token.h"
#include "google/cloud/internal/oauth2_credentials.h"
#include "google/cloud/options.h"
#include "google/cloud/status.h"
#include "google/cloud/status_or.h"
#include "google/cloud/storage/client.h"
#include "google/cloud/storage/client_options.h"
#include "google/cloud/storage/internal/storage_connection.h"

namespace array_record {
namespace {

using ::array_record::gcs_client_internal::SharedTokenCredentials;
using ::google::cloud::AccessToken;
using ::google::cloud::Status;
using ::google::cloud::StatusCode;
using ::google::cloud::StatusOr;
using ::google::cloud::storage::Client;
using ::google::cloud::storage::Oauth2CredentialsOption;
using ::google::cloud::storage::internal::ClientImplDetails;

// Token source that returns a fixed result.
class FakeTokenSource : public google::cloud::oauth2_internal::Credentials {
 public:
  explicit FakeTokenSource(StatusOr<AccessToken> token)
      : token_(std::move(token)) {}

  StatusOr<AccessToken> GetToken(
      std::chrono::system_clock::time_point) override {
    return token_;
  }

 private:
  StatusOr<AccessToken> token_;
};

TEST(SharedTokenCredentialsTest, ReturnsBearerHeader) {
  SharedTokenCredentials credentials(
      std::make_shared<FakeTokenSource>(AccessToken{
          "token-abc", absl::ToChronoTime(absl::Now() + absl::Hours(1))}));

  StatusOr<std::string> header = credentials.AuthorizationHeader();

  ASSERT_TRUE(header.ok()) << header.status();
  EXPECT_EQ(*header, "Authorization: Bearer token-abc");
}

TEST(SharedTokenCredentialsTest, ReturnsTokenError) {
  SharedTokenCredentials credentials(std::make_shared<FakeTokenSource>(
      Status(StatusCode::kUnavailable, "metadata server unreachable")));

  StatusOr<std::string> header = credentials.AuthorizationHeader();

  EXPECT_EQ(header.status().code(), StatusCode::kUnavailable);
  EXPECT_EQ(header.status().message(), "metadata server unreachable");
}

TEST(MakeGcsClientTest, ClientsShareOneTokenSource) {
  unsetenv("CLOUD_STORAGE_EMULATOR_ENDPOINT");
  unsetenv("CLOUD_STORAGE_TESTBENCH_ENDPOINT");

  Client first = MakeGcsClient();
  Client second = MakeGcsClient();

  // Each reader gets its own client, so they never share a curl handle pool.
  EXPECT_NE(first, second);
  google::cloud::Options first_options =
      ClientImplDetails::GetConnection(first)->options();
  google::cloud::Options second_options =
      ClientImplDetails::GetConnection(second)->options();
  ASSERT_TRUE(first_options.has<Oauth2CredentialsOption>());
  ASSERT_TRUE(second_options.has<Oauth2CredentialsOption>());
  // Both clients use the same credentials object, so they share its token.
  EXPECT_EQ(first_options.get<Oauth2CredentialsOption>(),
            second_options.get<Oauth2CredentialsOption>());
  EXPECT_NE(dynamic_cast<SharedTokenCredentials*>(
                first_options.get<Oauth2CredentialsOption>().get()),
            nullptr);
}

TEST(MakeGcsClientTest, EmulatorKeepsDefaultCredentials) {
  setenv("CLOUD_STORAGE_EMULATOR_ENDPOINT", "http://127.0.0.1:9", 1);

  Client client = MakeGcsClient();

  // Some google-cloud-cpp versions fill in insecure credentials for emulators,
  // so only check that the shared token isn't used.
  google::cloud::Options options =
      ClientImplDetails::GetConnection(client)->options();
  EXPECT_EQ(dynamic_cast<SharedTokenCredentials*>(
                options.get<Oauth2CredentialsOption>().get()),
            nullptr);
  unsetenv("CLOUD_STORAGE_EMULATOR_ENDPOINT");
}

}  // namespace
}  // namespace array_record

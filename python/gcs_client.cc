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

#include <stdlib.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "google/cloud/credentials.h"
#include "google/cloud/internal/oauth2_credentials.h"
#include "google/cloud/internal/oauth2_error_credentials.h"
#include "google/cloud/internal/opentelemetry.h"
#include "google/cloud/internal/rest_client.h"
#include "google/cloud/internal/rest_context.h"
#include "google/cloud/internal/rest_request.h"
#include "google/cloud/internal/rest_response.h"
#include "google/cloud/internal/tracing_rest_client.h"
#include "google/cloud/internal/unified_rest_credentials.h"
#include "google/cloud/opentelemetry_options.h"
#include "google/cloud/options.h"
#include "google/cloud/status_or.h"
#include "google/cloud/storage/client.h"
#include "google/cloud/storage/internal/connection_factory.h"
#include "google/cloud/storage/internal/generic_stub_factory.h"
#include "google/cloud/storage/internal/rest/stub.h"
#include "google/cloud/storage/options.h"

namespace array_record {
namespace gcs_client_internal {

using ::google::cloud::StatusOr;
using ::google::cloud::oauth2_internal::Credentials;
using ::google::cloud::rest_internal::RestClient;
using ::google::cloud::rest_internal::RestContext;
using ::google::cloud::rest_internal::RestRequest;
using ::google::cloud::rest_internal::RestResponse;

SharedCredentialsRestClient::SharedCredentialsRestClient(
    std::unique_ptr<RestClient> client, std::string endpoint,
    std::shared_ptr<Credentials> credentials)
    : client_(std::move(client)),
      endpoint_(std::move(endpoint)),
      credentials_(std::move(credentials)) {}

StatusOr<RestRequest> SharedCredentialsRestClient::Authorize(
    RestRequest const& request) {
  auto headers = credentials_->AuthenticationHeaders(
      absl::ToChronoTime(absl::Now()), endpoint_);
  if (!headers.ok()) return std::move(headers).status();
  RestRequest authorized = request;
  for (auto& header : *headers) authorized.AddHeader(std::move(header));
  return authorized;
}

StatusOr<std::unique_ptr<RestResponse>> SharedCredentialsRestClient::Delete(
    RestContext& context, RestRequest const& request) {
  auto authorized = Authorize(request);
  if (!authorized.ok()) return std::move(authorized).status();
  return client_->Delete(context, *authorized);
}

StatusOr<std::unique_ptr<RestResponse>> SharedCredentialsRestClient::Get(
    RestContext& context, RestRequest const& request) {
  auto authorized = Authorize(request);
  if (!authorized.ok()) return std::move(authorized).status();
  return client_->Get(context, *authorized);
}

StatusOr<std::unique_ptr<RestResponse>> SharedCredentialsRestClient::Patch(
    RestContext& context, RestRequest const& request,
    std::vector<absl::Span<char const>> const& payload) {
  auto authorized = Authorize(request);
  if (!authorized.ok()) return std::move(authorized).status();
  return client_->Patch(context, *authorized, payload);
}

StatusOr<std::unique_ptr<RestResponse>> SharedCredentialsRestClient::Post(
    RestContext& context, RestRequest const& request,
    std::vector<absl::Span<char const>> const& payload) {
  auto authorized = Authorize(request);
  if (!authorized.ok()) return std::move(authorized).status();
  return client_->Post(context, *authorized, payload);
}

StatusOr<std::unique_ptr<RestResponse>> SharedCredentialsRestClient::Post(
    RestContext& context, RestRequest const& request,
    std::vector<std::pair<std::string, std::string>> const& form_data) {
  auto authorized = Authorize(request);
  if (!authorized.ok()) return std::move(authorized).status();
  return client_->Post(context, *authorized, form_data);
}

StatusOr<std::unique_ptr<RestResponse>> SharedCredentialsRestClient::Put(
    RestContext& context, RestRequest const& request,
    std::vector<absl::Span<char const>> const& payload) {
  auto authorized = Authorize(request);
  if (!authorized.ok()) return std::move(authorized).status();
  return client_->Put(context, *authorized, payload);
}

namespace {

// Returns a transport for `endpoint` that authorizes its requests from
// `credentials`.
//
// The pooled client gets no credentials. Given credentials, it would add its
// own authentication headers and create its own token cache. It also gets no
// tracing. When tracing is on, the tracing client wraps
// `SharedCredentialsRestClient` instead, as it wraps `CurlRestClient` in
// `Client()`. So a span is made before the authentication headers are added
// and never records them, and a credentials error still gets a span.
std::unique_ptr<RestClient> MakeTransport(
    std::string const& endpoint, google::cloud::Options options,
    std::shared_ptr<Credentials> credentials) {
  bool const tracing_enabled = google::cloud::internal::TracingEnabled(options);
  options.unset<google::cloud::UnifiedCredentialsOption>();
  options.set<google::cloud::OpenTelemetryTracingOption>(false);
  std::unique_ptr<RestClient> client =
      std::make_unique<SharedCredentialsRestClient>(
          google::cloud::rest_internal::MakePooledRestClient(
              endpoint, std::move(options)),
          endpoint, std::move(credentials));
  if (tracing_enabled) {
    client =
        google::cloud::rest_internal::MakeTracingRestClient(std::move(client));
  }
  return client;
}

}  // namespace

google::cloud::storage::Client MakeGcsClient(
    std::shared_ptr<Credentials> credentials, google::cloud::Options options) {
  namespace storage = ::google::cloud::storage;
  namespace storage_internal = ::google::cloud::storage_internal;
  // The defaults include per-client Application Default Credentials. They stay
  // in the connection options, which only `Client::CreateV4SignedUrl()` and
  // similar signing calls read. The transports get no credentials, and
  // `SharedCredentialsRestClient` adds the authentication headers instead.
  options = storage::internal::DefaultOptions(std::move(options));
  std::string storage_endpoint = options.get<storage::RestEndpointOption>();
  std::string iam_endpoint = options.get<storage::IamEndpointOption>();
  std::shared_ptr<RestClient> storage_client = MakeTransport(
      storage_endpoint,
      storage::internal::RestStub::ResolveStorageAuthority(options),
      credentials);
  std::shared_ptr<RestClient> iam_client = MakeTransport(
      iam_endpoint, storage::internal::RestStub::ResolveIamAuthority(options),
      std::move(credentials));
  // Same stack as `Client(Options)`, except for the transports.
  return storage::internal::ClientImplDetails::CreateWithoutDecorations(
      storage_internal::MakeStorageConnection(
          options, storage_internal::DecorateStub(
                       options, std::make_unique<storage::internal::RestStub>(
                                    options, std::move(storage_client),
                                    std::move(iam_client)))));
}

std::shared_ptr<Credentials> SharedDefaultCredentials() {
  // Created on first use and never destroyed, so that readers on other threads
  // can still use them during shutdown.
  static absl::Mutex* const kMutex = new absl::Mutex;
  static std::shared_ptr<Credentials>* const kCredentials =
      new std::shared_ptr<Credentials>;
  absl::MutexLock lock(*kMutex);
  if (*kCredentials != nullptr) return *kCredentials;
  std::shared_ptr<Credentials> credentials =
      google::cloud::rest_internal::MapCredentials(
          *google::cloud::MakeGoogleDefaultCredentials());
  // Don't keep credentials that failed to load, for example because
  // `GOOGLE_APPLICATION_CREDENTIALS` names a file that isn't there yet. The
  // next reader tries again, as each reader did with its own client.
  if (dynamic_cast<google::cloud::oauth2_internal::ErrorCredentials*>(
          credentials.get()) != nullptr) {
    return credentials;
  }
  *kCredentials = credentials;
  return credentials;
}

}  // namespace gcs_client_internal

google::cloud::storage::Client MakeGcsClient() {
  // When a storage emulator is configured, google-cloud-cpp 3.x still uses
  // Application Default Credentials, even though `Client()` documents
  // otherwise. Outside of GCE they look up the metadata server and the request
  // is retried until the retry policy expires, so the reader appears to hang.
  // Use insecure credentials, as emulators expect.
  //
  // TODO(zhixiangli): Use `google::cloud::storage::Client()` again once
  // https://github.com/googleapis/google-cloud-cpp/issues/16678 is fixed in the
  // google-cloud-cpp release used by the OSS build.
  if (getenv("CLOUD_STORAGE_EMULATOR_ENDPOINT") != nullptr ||
      getenv("CLOUD_STORAGE_TESTBENCH_ENDPOINT") != nullptr) {
    return google::cloud::storage::Client(
        google::cloud::Options{}.set<google::cloud::UnifiedCredentialsOption>(
            google::cloud::MakeInsecureCredentials()));
  }
  return gcs_client_internal::MakeGcsClient(
      gcs_client_internal::SharedDefaultCredentials(),
      google::cloud::Options{});
}

}  // namespace array_record

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

#ifndef ARRAY_RECORD_PYTHON_GCS_CLIENT_H_
#define ARRAY_RECORD_PYTHON_GCS_CLIENT_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "google/cloud/internal/oauth2_credentials.h"
#include "google/cloud/internal/rest_client.h"
#include "google/cloud/internal/rest_context.h"
#include "google/cloud/internal/rest_request.h"
#include "google/cloud/internal/rest_response.h"
#include "google/cloud/options.h"
#include "google/cloud/status_or.h"
#include "google/cloud/storage/client.h"

namespace array_record {

// Returns a new GCS client for one reader.
//
// Readers don't share a client: concurrent use of one client's curl handle
// pool hits a libcurl data race (Curl_peer_unlink). Instead, every client
// authorizes its requests from one process-wide credentials object, so a
// process fetches and refreshes its access token once instead of once per
// reader.
//
// When `CLOUD_STORAGE_EMULATOR_ENDPOINT` or `CLOUD_STORAGE_TESTBENCH_ENDPOINT`
// is set, the client uses insecure credentials, as emulators expect.
//
// Limitations:
// - Each reader still opens its own HTTP connections. Only the credentials are
//   shared.
// - This uses google-cloud-cpp internal APIs (`rest_internal::RestClient`,
//   `rest_internal::MakePooledRestClient`, `rest_internal::MapCredentials`,
//   `oauth2_internal::Credentials`, `storage::internal::RestStub`,
//   `storage_internal::DecorateStub`, `storage_internal::MakeStorageConnection`
//   and `storage::internal::ClientImplDetails`), because the public API can't
//   share credentials between clients. Internal APIs can change in any
//   release. gcs_client_test breaks if they do. Replace this once
//   google-cloud-cpp offers a public way to share credentials, or once one
//   client can be shared.
google::cloud::storage::Client MakeGcsClient();

namespace gcs_client_internal {

// Wraps a `RestClient` and adds the authentication headers from shared
// credentials to every request before forwarding it.
//
// These are the same headers that `CurlRestClient` adds from
// `UnifiedCredentialsOption`: `Authorization` (omitted for an empty token)
// and, for credentials that support it, `x-allowed-locations`. The wrapped
// client must not add its own authentication headers.
class SharedCredentialsRestClient
    : public google::cloud::rest_internal::RestClient {
 public:
  SharedCredentialsRestClient(
      std::unique_ptr<google::cloud::rest_internal::RestClient> client,
      std::string endpoint,
      std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials);

  google::cloud::StatusOr<
      std::unique_ptr<google::cloud::rest_internal::RestResponse>>
  Delete(google::cloud::rest_internal::RestContext& context,
         google::cloud::rest_internal::RestRequest const& request) override;
  google::cloud::StatusOr<
      std::unique_ptr<google::cloud::rest_internal::RestResponse>>
  Get(google::cloud::rest_internal::RestContext& context,
      google::cloud::rest_internal::RestRequest const& request) override;
  google::cloud::StatusOr<
      std::unique_ptr<google::cloud::rest_internal::RestResponse>>
  Patch(google::cloud::rest_internal::RestContext& context,
        google::cloud::rest_internal::RestRequest const& request,
        std::vector<absl::Span<char const>> const& payload) override;
  google::cloud::StatusOr<
      std::unique_ptr<google::cloud::rest_internal::RestResponse>>
  Post(google::cloud::rest_internal::RestContext& context,
       google::cloud::rest_internal::RestRequest const& request,
       std::vector<absl::Span<char const>> const& payload) override;
  google::cloud::StatusOr<
      std::unique_ptr<google::cloud::rest_internal::RestResponse>>
  Post(google::cloud::rest_internal::RestContext& context,
       google::cloud::rest_internal::RestRequest const& request,
       std::vector<std::pair<std::string, std::string>> const& form_data)
      override;
  google::cloud::StatusOr<
      std::unique_ptr<google::cloud::rest_internal::RestResponse>>
  Put(google::cloud::rest_internal::RestContext& context,
      google::cloud::rest_internal::RestRequest const& request,
      std::vector<absl::Span<char const>> const& payload) override;

 private:
  // Returns a copy of `request` with the authentication headers added.
  google::cloud::StatusOr<google::cloud::rest_internal::RestRequest> Authorize(
      google::cloud::rest_internal::RestRequest const& request);

  std::unique_ptr<google::cloud::rest_internal::RestClient> client_;
  std::string endpoint_;
  std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials_;
};

// Returns a new GCS client that authorizes its requests from `credentials`.
// `options` override the storage defaults, except that the transports never
// get credentials of their own. `MakeGcsClient()` calls this with the
// process-wide credentials and no overrides. Tests call it with their own
// credentials and endpoint.
google::cloud::storage::Client MakeGcsClient(
    std::shared_ptr<google::cloud::oauth2_internal::Credentials> credentials,
    google::cloud::Options options);

// Returns the process-wide Application Default Credentials that
// `MakeGcsClient()` uses. Every call returns the same object once they load.
// Credentials that fail to load aren't kept, so the next call loads them again.
std::shared_ptr<google::cloud::oauth2_internal::Credentials>
SharedDefaultCredentials();

}  // namespace gcs_client_internal
}  // namespace array_record

#endif  // ARRAY_RECORD_PYTHON_GCS_CLIENT_H_

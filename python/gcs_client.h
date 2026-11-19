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

#include "google/cloud/internal/oauth2_credentials.h"
#include "google/cloud/status_or.h"
#include "google/cloud/storage/client.h"
#include "google/cloud/storage/oauth2/credentials.h"

namespace array_record {

// Returns a new GCS client for one reader.
//
// Readers don't share a client: concurrent use of one client's curl handle
// pool hits a libcurl data race (Curl_peer_unlink). Instead, all clients share
// one cached access token, so a process fetches its token once instead of once
// per reader.
//
// Limitation: each reader still opens its own HTTP connections, because only
// the token is shared.
google::cloud::storage::Client MakeGcsClient();

namespace gcs_client_internal {

// Authorizes requests with a token from a shared token source.
//
// Limitation: this uses google-cloud-cpp internal APIs
// (`oauth2_internal::Credentials`, `rest_internal::MapCredentials`) and the
// deprecated `storage::oauth2::Credentials`, because the public API can't share
// a token source between clients. Internal APIs can change in any release.
// gcs_client_test breaks if they do. Replace this class once google-cloud-cpp
// offers a public way to share credentials, or once one client can be shared.
class SharedTokenCredentials
    : public google::cloud::storage::oauth2::Credentials {
 public:
  explicit SharedTokenCredentials(
      std::shared_ptr<google::cloud::oauth2_internal::Credentials> source);

  google::cloud::StatusOr<std::string> AuthorizationHeader() override;

 private:
  std::shared_ptr<google::cloud::oauth2_internal::Credentials> source_;
};

}  // namespace gcs_client_internal
}  // namespace array_record

#endif  // ARRAY_RECORD_PYTHON_GCS_CLIENT_H_

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

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "google/cloud/credentials.h"
#include "google/cloud/internal/oauth2_credentials.h"
#include "google/cloud/internal/unified_rest_credentials.h"
#include "google/cloud/options.h"
#include "google/cloud/status_or.h"
#include "google/cloud/storage/client.h"
#include "google/cloud/storage/client_options.h"
#include "google/cloud/storage/oauth2/credentials.h"

namespace array_record {
namespace gcs_client_internal {

SharedTokenCredentials::SharedTokenCredentials(
    std::shared_ptr<google::cloud::oauth2_internal::Credentials> source)
    : source_(std::move(source)) {}

google::cloud::StatusOr<std::string>
SharedTokenCredentials::AuthorizationHeader() {
  auto token = source_->GetToken(absl::ToChronoTime(absl::Now()));
  if (!token) return std::move(token).status();
  return absl::StrCat("Authorization: Bearer ", token->token);
}

}  // namespace gcs_client_internal

google::cloud::storage::Client MakeGcsClient() {
  // An emulator takes no token. Keep the client's default insecure credentials.
  if (std::getenv("CLOUD_STORAGE_EMULATOR_ENDPOINT") != nullptr ||
      std::getenv("CLOUD_STORAGE_TESTBENCH_ENDPOINT") != nullptr) {
    return google::cloud::storage::Client();
  }
  // The token source caches the token and refreshes it before it expires.
  // Never destroyed, because readers can outlive static destruction.
  static const auto* const kCredentials =
      new std::shared_ptr<google::cloud::storage::oauth2::Credentials>(
          std::make_shared<gcs_client_internal::SharedTokenCredentials>(
              google::cloud::rest_internal::MapCredentials(
                  *google::cloud::MakeGoogleDefaultCredentials())));
  return google::cloud::storage::Client(
      google::cloud::Options{}
          .set<google::cloud::storage::Oauth2CredentialsOption>(*kCredentials));
}

}  // namespace array_record

# ArrayRecord is a new file format for IO intensive applications.
# It supports efficient random access and various compression algorithms.

load("@rules_proto//proto:defs.bzl", "proto_library")
load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

proto_library(
    name = "layout_proto",
    srcs = ["layout.proto"],
)

cc_proto_library(
    name = "layout_cc_proto",
    deps = [":layout_proto"],
)

cc_library(
    name = "common",
    hdrs = ["common.h"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

cc_library(
    name = "sequenced_chunk_writer",
    srcs = ["sequenced_chunk_writer.cc"],
    hdrs = ["sequenced_chunk_writer.h"],
    deps = [
        ":common",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@riegeli//riegeli/base:initializer",
        "@riegeli//riegeli/base:object",
        "@riegeli//riegeli/base:status",
        "@riegeli//riegeli/base:types",
        "@riegeli//riegeli/bytes:writer",
        "@riegeli//riegeli/chunk_encoding:chunk",
        "@riegeli//riegeli/chunk_encoding:constants",
        "@riegeli//riegeli/records:chunk_writer",
    ],
)

cc_library(
    name = "thread_pool",
    srcs = ["thread_pool.cc"],
    hdrs = ["thread_pool.h"],
    deps = [
        "@abseil-cpp//absl/flags:flag",
        "@eigen//:eigen",
    ],
)

cc_library(
    name = "parallel_for",
    hdrs = ["parallel_for.h"],
    deps = [
        ":common",
        ":thread_pool",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
    ],
)

cc_library(
    name = "tri_state_ptr",
    hdrs = ["tri_state_ptr.h"],
    deps = [
        ":common",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/synchronization",
    ],
)

cc_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [":common"],
)

cc_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.cc"],
    deps = [
        ":common",
        ":test_utils",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "array_record_writer",
    srcs = ["array_record_writer.cc"],
    hdrs = ["array_record_writer.h"],
    deps = [
        ":common",
        ":layout_cc_proto",
        ":sequenced_chunk_writer",
        ":thread_pool",
        ":tri_state_ptr",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@protobuf//:protobuf_lite",
        "@riegeli//riegeli/base:initializer",
        "@riegeli//riegeli/base:object",
        "@riegeli//riegeli/base:options_parser",
        "@riegeli//riegeli/base:status",
        "@riegeli//riegeli/bytes:chain_writer",
        "@riegeli//riegeli/bytes:writer",
        "@riegeli//riegeli/chunk_encoding:chunk",
        "@riegeli//riegeli/chunk_encoding:chunk_encoder",
        "@riegeli//riegeli/chunk_encoding:compressor_options",
        "@riegeli//riegeli/chunk_encoding:constants",
        "@riegeli//riegeli/chunk_encoding:deferred_encoder",
        "@riegeli//riegeli/chunk_encoding:simple_encoder",
        "@riegeli//riegeli/chunk_encoding:transpose_encoder",
        "@riegeli//riegeli/records:records_metadata_cc_proto",
    ],
)

cc_library(
    name = "masked_reader",
    srcs = ["masked_reader.cc"],
    hdrs = ["masked_reader.h"],
    deps = [
        ":common",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:optional",
        "@riegeli//riegeli/base:object",
        "@riegeli//riegeli/base:status",
        "@riegeli//riegeli/base:types",
        "@riegeli//riegeli/bytes:reader",
    ],
)

cc_library(
    name = "array_record_reader",
    srcs = ["array_record_reader.cc"],
    hdrs = ["array_record_reader.h"],
    deps = [
        ":common",
        ":layout_cc_proto",
        ":masked_reader",
        ":parallel_for",
        ":thread_pool",
        ":tri_state_ptr",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@protobuf//:protobuf_lite",
        "@riegeli//riegeli/base:initializer",
        "@riegeli//riegeli/base:object",
        "@riegeli//riegeli/base:options_parser",
        "@riegeli//riegeli/base:status",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/chunk_encoding:chunk",
        "@riegeli//riegeli/chunk_encoding:chunk_decoder",
        "@riegeli//riegeli/records:chunk_reader",
    ],
)

cc_test(
    name = "sequenced_chunk_writer_test",
    srcs = ["sequenced_chunk_writer_test.cc"],
    deps = [
        ":common",
        ":sequenced_chunk_writer",
        ":thread_pool",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/base:initializer",
        "@riegeli//riegeli/base:shared_ptr",
        "@riegeli//riegeli/bytes:chain_writer",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/bytes:string_writer",
        "@riegeli//riegeli/chunk_encoding:chunk",
        "@riegeli//riegeli/chunk_encoding:compressor_options",
        "@riegeli//riegeli/chunk_encoding:constants",
        "@riegeli//riegeli/chunk_encoding:simple_encoder",
        "@riegeli//riegeli/records:record_reader",
    ],
)

cc_test(
    name = "tri_state_ptr_test",
    srcs = ["tri_state_ptr_test.cc"],
    deps = [
        ":common",
        ":thread_pool",
        ":tri_state_ptr",
        "@abseil-cpp//absl/synchronization",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/base:initializer",
    ],
)

cc_test(
    name = "array_record_writer_test",
    srcs = ["array_record_writer_test.cc"],
    shard_count = 4,
    tags = ["notsan"],
    deps = [
        ":array_record_writer",
        ":common",
        ":layout_cc_proto",
        ":test_utils",
        ":thread_pool",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:cord_test_helpers",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/base:initializer",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/bytes:string_writer",
        "@riegeli//riegeli/chunk_encoding:constants",
        "@riegeli//riegeli/records:record_reader",
        "@riegeli//riegeli/records:records_metadata_cc_proto",
    ],
)

cc_test(
    name = "masked_reader_test",
    srcs = ["masked_reader_test.cc"],
    deps = [
        ":masked_reader",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:string_reader",
    ],
)

cc_test(
    name = "parallel_for_test",
    size = "small",
    srcs = ["parallel_for_test.cc"],
    deps = [
        ":common",
        ":parallel_for",
        ":thread_pool",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "array_record_reader_test",
    srcs = ["array_record_reader_test.cc"],
    shard_count = 4,
    deps = [
        ":array_record_reader",
        ":array_record_writer",
        ":common",
        ":layout_cc_proto",
        ":test_utils",
        ":thread_pool",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/base:initializer",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/bytes:string_writer",
    ],
)

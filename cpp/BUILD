# ArrayRecord is a new file format for IO intensive applications.
# It supports efficient random access and various compression algorithms.

load("@rules_proto//proto:defs.bzl", "proto_library")

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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "sequenced_chunk_writer",
    srcs = ["sequenced_chunk_writer.cc"],
    hdrs = ["sequenced_chunk_writer.h"],
    deps = [
        ":common",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_riegeli//riegeli/base:initializer",
        "@com_google_riegeli//riegeli/base:object",
        "@com_google_riegeli//riegeli/base:status",
        "@com_google_riegeli//riegeli/base:types",
        "@com_google_riegeli//riegeli/bytes:writer",
        "@com_google_riegeli//riegeli/chunk_encoding:chunk",
        "@com_google_riegeli//riegeli/chunk_encoding:constants",
        "@com_google_riegeli//riegeli/records:chunk_writer",
    ],
)

cc_library(
    name = "thread_pool",
    srcs = ["thread_pool.cc"],
    hdrs = ["thread_pool.h"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@eigen3//:eigen3",
    ],
)

cc_library(
    name = "parallel_for",
    hdrs = ["parallel_for.h"],
    deps = [
        ":common",
        ":thread_pool",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "tri_state_ptr",
    hdrs = ["tri_state_ptr.h"],
    deps = [
        ":common",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
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
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf_lite",
        "@com_google_riegeli//riegeli/base:initializer",
        "@com_google_riegeli//riegeli/base:object",
        "@com_google_riegeli//riegeli/base:options_parser",
        "@com_google_riegeli//riegeli/base:status",
        "@com_google_riegeli//riegeli/bytes:chain_writer",
        "@com_google_riegeli//riegeli/bytes:writer",
        "@com_google_riegeli//riegeli/chunk_encoding:chunk",
        "@com_google_riegeli//riegeli/chunk_encoding:chunk_encoder",
        "@com_google_riegeli//riegeli/chunk_encoding:compressor_options",
        "@com_google_riegeli//riegeli/chunk_encoding:constants",
        "@com_google_riegeli//riegeli/chunk_encoding:deferred_encoder",
        "@com_google_riegeli//riegeli/chunk_encoding:simple_encoder",
        "@com_google_riegeli//riegeli/chunk_encoding:transpose_encoder",
        "@com_google_riegeli//riegeli/records:records_metadata_cc_proto",
    ],
)

cc_library(
    name = "masked_reader",
    srcs = ["masked_reader.cc"],
    hdrs = ["masked_reader.h"],
    deps = [
        ":common",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_riegeli//riegeli/base:object",
        "@com_google_riegeli//riegeli/base:status",
        "@com_google_riegeli//riegeli/base:types",
        "@com_google_riegeli//riegeli/bytes:reader",
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf_lite",
        "@com_google_riegeli//riegeli/base:initializer",
        "@com_google_riegeli//riegeli/base:object",
        "@com_google_riegeli//riegeli/base:options_parser",
        "@com_google_riegeli//riegeli/base:status",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/chunk_encoding:chunk",
        "@com_google_riegeli//riegeli/chunk_encoding:chunk_decoder",
        "@com_google_riegeli//riegeli/records:chunk_reader",
    ],
)

cc_test(
    name = "sequenced_chunk_writer_test",
    srcs = ["sequenced_chunk_writer_test.cc"],
    deps = [
        ":common",
        ":sequenced_chunk_writer",
        ":thread_pool",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/base:initializer",
        "@com_google_riegeli//riegeli/base:shared_ptr",
        "@com_google_riegeli//riegeli/bytes:chain_writer",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:string_reader",
        "@com_google_riegeli//riegeli/bytes:string_writer",
        "@com_google_riegeli//riegeli/chunk_encoding:chunk",
        "@com_google_riegeli//riegeli/chunk_encoding:compressor_options",
        "@com_google_riegeli//riegeli/chunk_encoding:constants",
        "@com_google_riegeli//riegeli/chunk_encoding:simple_encoder",
        "@com_google_riegeli//riegeli/records:record_reader",
    ],
)

cc_test(
    name = "tri_state_ptr_test",
    srcs = ["tri_state_ptr_test.cc"],
    deps = [
        ":common",
        ":thread_pool",
        ":tri_state_ptr",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/base:initializer",
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
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/base:initializer",
        "@com_google_riegeli//riegeli/bytes:string_reader",
        "@com_google_riegeli//riegeli/bytes:string_writer",
        "@com_google_riegeli//riegeli/chunk_encoding:constants",
        "@com_google_riegeli//riegeli/records:record_reader",
        "@com_google_riegeli//riegeli/records:records_metadata_cc_proto",
    ],
)

cc_test(
    name = "masked_reader_test",
    srcs = ["masked_reader_test.cc"],
    deps = [
        ":masked_reader",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:string_reader",
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
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
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
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/base:initializer",
        "@com_google_riegeli//riegeli/bytes:string_reader",
        "@com_google_riegeli//riegeli/bytes:string_writer",
    ],
)

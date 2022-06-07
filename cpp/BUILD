# ArrayRecord is a new file format for IO intensive applications.
# It supports efficient random access and various compression algorithms.

load("//security/fuzzing/blaze:cc_fuzz_target.bzl", "cc_fuzz_target")

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
        "//third_party/absl/base:core_headers",
        "//third_party/absl/status",
        "//third_party/absl/strings:str_format",
    ],
)

cc_library(
    name = "sequenced_chunk_writer",
    srcs = ["sequenced_chunk_writer.cc"],
    hdrs = ["sequenced_chunk_writer.h"],
    deps = [
        ":common",
        "//third_party/absl/base:core_headers",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/absl/synchronization",
        "//third_party/riegeli/base",
        "//third_party/riegeli/base:status",
        "//third_party/riegeli/bytes:writer",
        "//third_party/riegeli/chunk_encoding:chunk",
        "//third_party/riegeli/chunk_encoding:constants",
        "//third_party/riegeli/records:chunk_writer",
    ],
)

cc_library(
    name = "thread_pool",
    srcs = ["thread_pool.cc"],
    hdrs = ["thread_pool.h"],
    deps = [
        "//third_party/absl/flags:flag",
        "@eigen3//:eigen3",
    ],
)

cc_library(
    name = "parallel_for",
    hdrs = ["parallel_for.h"],
    deps = [
        ":common",
        ":thread_pool",
        "//third_party/absl/base:core_headers",
        "//third_party/absl/functional:function_ref",
        "//third_party/absl/status",
        "//third_party/absl/synchronization",
    ],
)

cc_library(
    name = "thread_compatible_shared_ptr",
    hdrs = ["thread_compatible_shared_ptr.h"],
    deps = ["//third_party/absl/synchronization"],
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
        "//testing/base/public:gunit_main",
        "//third_party/absl/strings",
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
        "//third_party/absl/base:core_headers",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/synchronization",
        "//third_party/absl/types:span",
        "//third_party/riegeli/base",
        "//third_party/riegeli/base:options_parser",
        "//third_party/riegeli/bytes:chain_writer",
        "//third_party/riegeli/bytes:file_writer",
        "//third_party/riegeli/chunk_encoding:chunk",
        "//third_party/riegeli/chunk_encoding:chunk_encoder",
        "//third_party/riegeli/chunk_encoding:compressor_options",
        "//third_party/riegeli/chunk_encoding:constants",
        "//third_party/riegeli/chunk_encoding:deferred_encoder",
        "//third_party/riegeli/chunk_encoding:simple_encoder",
        "//third_party/riegeli/chunk_encoding:transpose_encoder",
        "//third_party/riegeli/records:records_metadata_cc_proto",
        "@com_google_protobuf//:protobuf_lite",
    ],
)

cc_library(
    name = "masked_reader",
    srcs = ["masked_reader.cc"],
    hdrs = ["masked_reader.h"],
    deps = [
        ":common",
        "//third_party/absl/memory",
        "//third_party/absl/status",
        "//third_party/absl/time",
        "//third_party/absl/types:optional",
        "//third_party/riegeli/base",
        "//third_party/riegeli/base:status",
        "//third_party/riegeli/bytes:reader",
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
        ":thread_compatible_shared_ptr",
        ":thread_pool",
        "//third_party/absl/base:core_headers",
        "//third_party/absl/functional:bind_front",
        "//third_party/absl/functional:function_ref",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/absl/synchronization",
        "//third_party/absl/types:span",
        "//third_party/riegeli/base",
        "//third_party/riegeli/base:options_parser",
        "//third_party/riegeli/base:status",
        "//third_party/riegeli/bytes:reader",
        "//third_party/riegeli/chunk_encoding:chunk_decoder",
        "//third_party/riegeli/records:chunk_reader",
        "//third_party/riegeli/records:record_position",
        "//third_party/riegeli/records:record_reader",
        "@com_google_protobuf//:protobuf_lite",
    ],
)

cc_test(
    name = "sequenced_chunk_writer_test",
    srcs = ["sequenced_chunk_writer_test.cc"],
    deps = [
        ":common",
        ":sequenced_chunk_writer",
        ":thread_pool",
        "//testing/base/public:gunit_main",
        "//third_party/absl/flags:flag",
        "//third_party/absl/status",
        "//third_party/absl/strings:cord",
        "//third_party/riegeli/bytes:chain_writer",
        "//third_party/riegeli/bytes:cord_writer",
        "//third_party/riegeli/bytes:string_reader",
        "//third_party/riegeli/bytes:string_writer",
        "//third_party/riegeli/chunk_encoding:chunk",
        "//third_party/riegeli/chunk_encoding:compressor_options",
        "//third_party/riegeli/chunk_encoding:simple_encoder",
        "//third_party/riegeli/records:record_reader",
    ],
)

cc_test(
    name = "thread_compatible_shared_ptr_test",
    srcs = ["thread_compatible_shared_ptr_test.cc"],
    deps = [
        ":common",
        ":thread_compatible_shared_ptr",
        ":thread_pool",
        "//testing/base/public:gunit_main",
        "//third_party/absl/time",
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
        "//testing/base/public:gunit_main",
        "//third_party/absl/strings",
        "//third_party/riegeli/bytes:string_reader",
        "//third_party/riegeli/bytes:string_writer",
        "//third_party/riegeli/chunk_encoding:constants",
        "//third_party/riegeli/records:record_reader",
        "//third_party/riegeli/records:record_writer",
        "//third_party/riegeli/records:records_metadata_cc_proto",
    ],
)

cc_test(
    name = "masked_reader_test",
    srcs = ["masked_reader_test.cc"],
    deps = [
        ":masked_reader",
        "//testing/base/public:gunit_main",
        "//third_party/riegeli/bytes:string_reader",
    ],
)

cc_test(
    name = "parallel_for_test",
    size = "small",
    srcs = ["parallel_for_test.cc"],
    deps = [
        ":parallel_for",
        ":thread_pool",
        "//testing/base/public:gunit_main",
        "//third_party/absl/functional:function_ref",
        "//third_party/absl/status",
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
        "//testing/base/public:gunit_main",
        "//third_party/absl/functional:function_ref",
        "//third_party/absl/status",
        "//third_party/absl/strings",
        "//third_party/riegeli/bytes:string_reader",
        "//third_party/riegeli/bytes:string_writer",
        "//third_party/riegeli/chunk_encoding:chunk_decoder",
        "//third_party/riegeli/records:chunk_reader",
    ],
)

cc_fuzz_target(
    name = "array_record_fuzzer",
    srcs = ["array_record_fuzzer.cc"],
    componentid = 85655,  # Buganizer ID: Research > ScaM
    deps = [
        ":array_record_reader",
        ":array_record_writer",
        ":layout_cc_proto",
        ":thread_pool",
        "//base:logging",
        "//base:sysinfo",
        "//research/scam/utils:threads",
        "//third_party/absl/status",
        "//third_party/absl/types:span",
        "//third_party/riegeli/bytes:string_reader",
        "//third_party/riegeli/bytes:string_writer",
        "//third_party/riegeli/records:record_reader",
        "//third_party/riegeli/records:records_metadata_cc_proto",
    ],
)

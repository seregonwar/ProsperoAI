# ProsperoAI — Prospero Protocol (whitepaper §24/§25)
#
# Transport-independent binary protocol library for desktop <-> PS5
# communication. The protocol core is pure C and transport-agnostic; an
# in-memory pipe transport ships for host tests and loopback.
pai_define_library(pai_protocol
  crc32.c
  frame.c
  message.c
  connection.c
  transport_pipe.c
)

target_include_directories(pai_protocol PUBLIC ${PROJECT_SOURCE_DIR})

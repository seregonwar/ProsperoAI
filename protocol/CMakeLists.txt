# ProsperoAI — Prospero Protocol (whitepaper §24/§25)
#
# Transport-independent binary protocol library for desktop <-> PS5
# communication. The protocol core is pure C and transport-agnostic; an
# in-memory pipe transport ships for host tests and loopback. The TCP
# transport is host-side (the console payload uses its own sockets).
set(PAI_PROTOCOL_SOURCES
  crc32.c
  frame.c
  message.c
  connection.c
  transport_pipe.c
)
if(PAI_PLATFORM STREQUAL "host")
  list(APPEND PAI_PROTOCOL_SOURCES transport_tcp.c ping.c)
endif()

pai_define_library(pai_protocol ${PAI_PROTOCOL_SOURCES})

target_include_directories(pai_protocol PUBLIC ${PROJECT_SOURCE_DIR})
if(WIN32 AND PAI_PLATFORM STREQUAL "host")
  target_link_libraries(pai_protocol PRIVATE ws2_32)
endif()

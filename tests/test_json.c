#include "test.h"

#include <gateway/json.h>

#include <string.h>

static int
parse_ok(pai_json_doc_t *doc, const char *s) {
  return pai_json_parse(doc, s, (uint32_t)strlen(s)) == PAI_OK;
}

TEST_MAIN_BEGIN()

/* Basic object access. */
{
  pai_json_doc_t doc;
  int32_t root, m;
  CHECK(parse_ok(&doc, "{\"a\":1,\"b\":2.5,\"c\":\"hi\",\"d\":true,"
                       "\"e\":null,\"f\":[1,2,3]}"));
  root = pai_json_root(&doc);
  CHECK(pai_json_type(&doc, root) == PAI_JSON_OBJECT);
  m = pai_json_member(&doc, root, "a");
  CHECK(pai_json_type(&doc, m) == PAI_JSON_NUMBER);
  CHECK_EQ_INT((int)pai_json_num(&doc, m), 1);
  CHECK(pai_json_type(&doc, pai_json_member(&doc, root, "b")) ==
        PAI_JSON_NUMBER);
  CHECK_EQ_UINT((unsigned)(pai_json_num(&doc, pai_json_member(&doc, root, "b")) * 10),
                25);
  CHECK(strcmp(pai_json_str(&doc, pai_json_member(&doc, root, "c")), "hi") ==
        0);
  CHECK(pai_json_bool(&doc, pai_json_member(&doc, root, "d")) == 1);
  CHECK(pai_json_type(&doc, pai_json_member(&doc, root, "e")) ==
        PAI_JSON_NULL);
  CHECK(pai_json_member(&doc, root, "missing") == -1);
  CHECK(strcmp(pai_json_str_member(&doc, root, "c"), "hi") == 0);
  CHECK(pai_json_str_member(&doc, root, "missing") == NULL);

  m = pai_json_member(&doc, root, "f");
  CHECK(pai_json_type(&doc, m) == PAI_JSON_ARRAY);
  CHECK_EQ_INT(pai_json_array_len(&doc, m), 3);
  CHECK_EQ_UINT((unsigned)pai_json_num(&doc, pai_json_array_at(&doc, m, 2)),
                3);
  CHECK(pai_json_array_at(&doc, m, 9) == -1);
  pai_json_destroy(&doc);
}

/* Nested structures + whitespace. */
{
  pai_json_doc_t doc;
  int32_t root, arr;
  CHECK(parse_ok(&doc, "  { \"x\" : [ { \"y\" : [1, 2] }, 3 ] }  "));
  root = pai_json_root(&doc);
  arr = pai_json_member(&doc, root, "x");
  CHECK_EQ_INT(pai_json_array_len(&doc, arr), 2);
  CHECK(pai_json_type(&doc, pai_json_array_at(&doc, arr, 1)) ==
        PAI_JSON_NUMBER);
  {
    int32_t inner = pai_json_array_at(&doc, arr, 0);
    CHECK_EQ_UINT((unsigned)pai_json_num(&doc,
                                         pai_json_member(&doc, inner, "y") != -1
                                             ? pai_json_array_at(
                                                   &doc, pai_json_member(&doc, inner, "y"), 1)
                                             : -1),
                  2);
  }
  pai_json_destroy(&doc);
}

/* Escapes. */
{
  pai_json_doc_t doc;
  int32_t root;
  CHECK(parse_ok(&doc, "{\"s\":\"a\\nb\\t\\\"q\\\\r\\/f\\u0041\"}"));
  root = pai_json_root(&doc);
  CHECK(strcmp(pai_json_str_member(&doc, root, "s"),
               "a\nb\t\"q\\r/fA") == 0);
  pai_json_destroy(&doc);

  /* Surrogate pair -> UTF-8 emoji. */
  CHECK(parse_ok(&doc, "{\"e\":\"\\uD83D\\uDE00\"}"));
  root = pai_json_root(&doc);
  CHECK(strcmp(pai_json_str_member(&doc, root, "e"), "\xF0\x9F\x98\x80") == 0);
  pai_json_destroy(&doc);

  /* Lone surrogate -> replacement char. */
  CHECK(parse_ok(&doc, "{\"e\":\"\\uD800x\"}"));
  root = pai_json_root(&doc);
  CHECK(strcmp(pai_json_str_member(&doc, root, "e"), "\xEF\xBF\xBDx") == 0);
  pai_json_destroy(&doc);
}

/* Numbers. */
{
  pai_json_doc_t doc;
  int32_t root, m;
  CHECK(parse_ok(&doc, "[0,-1,3.14,1e3,2.5e-2,-0]"));
  root = pai_json_root(&doc);
  CHECK_EQ_UINT((unsigned)pai_json_num(&doc, pai_json_array_at(&doc, root, 0)),
                0);
  CHECK(pai_json_num(&doc, pai_json_array_at(&doc, root, 1)) == -1.0);
  CHECK_EQ_INT((int)(pai_json_num(&doc, pai_json_array_at(&doc, root, 2)) * 100),
               314);
  CHECK_EQ_UINT((unsigned)pai_json_num(&doc, pai_json_array_at(&doc, root, 3)),
                1000);
  CHECK_EQ_INT((int)(pai_json_num(&doc, pai_json_array_at(&doc, root, 4)) * 1000),
               25);
  {
    double v = pai_json_num(&doc, pai_json_array_at(&doc, root, 5));
    CHECK(v == 0.0);
  }
  m = -1;
  (void)m;
  pai_json_destroy(&doc);
}

/* Serialization round-trip (semantic equality). */
{
  pai_json_doc_t doc, doc2;
  pai_json_wb_t wb;
  int32_t root, root2;
  CHECK(parse_ok(&doc, "{\"a\":[1,2.5,\"x\\ny\"],\"b\":{\"c\":true}}"));
  root = pai_json_root(&doc);
  pai_json_wb_init(&wb);
  CHECK(pai_json_serialize(&doc, root, &wb) == PAI_OK);
  CHECK(pai_json_wb_putc(&wb, '\0') == PAI_OK);
  CHECK(parse_ok(&doc2, wb.buf));
  root2 = pai_json_root(&doc2);
  CHECK(pai_json_type(&doc2, root2) == PAI_JSON_OBJECT);
  {
    int32_t a = pai_json_member(&doc2, root2, "a");
    CHECK_EQ_INT(pai_json_array_len(&doc2, a), 3);
    CHECK_EQ_INT((int)(pai_json_num(&doc2, pai_json_array_at(&doc2, a, 1)) * 10),
                 25);
    CHECK(strcmp(pai_json_str(&doc2, pai_json_array_at(&doc2, a, 2)),
                 "x\ny") == 0);
    {
      int32_t b = pai_json_member(&doc2, root2, "b");
      CHECK(pai_json_bool(&doc2, pai_json_member(&doc2, b, "c")) == 1);
    }
  }
  pai_json_wb_destroy(&wb);
  pai_json_destroy(&doc2);
  pai_json_destroy(&doc);
}

/* Quote escaping. */
{
  pai_json_wb_t wb;
  pai_json_wb_init(&wb);
  CHECK(pai_json_quote(&wb, "a\"b\\c\nd\x01", 8) == PAI_OK);
  CHECK(pai_json_wb_putc(&wb, '\0') == PAI_OK);
  CHECK(strcmp(wb.buf, "\"a\\\"b\\\\c\\nd\\u0001\"") == 0);
  pai_json_wb_destroy(&wb);
}

/* Malformed input. */
{
  pai_json_doc_t doc;
  CHECK(!parse_ok(&doc, "{}x"));             /* trailing garbage */
  CHECK(!parse_ok(&doc, "{\"a\":}"));        /* missing value    */
  CHECK(!parse_ok(&doc, "{\"a\"}"));         /* missing colon    */
  CHECK(!parse_ok(&doc, "{\"a\":1"));        /* unclosed object  */
  CHECK(!parse_ok(&doc, "[1,2"));            /* unclosed array   */
  CHECK(!parse_ok(&doc, "\"\\q\""));         /* bad escape       */
  CHECK(!parse_ok(&doc, "\"a\x01b\""));      /* raw control char */
  CHECK(!parse_ok(&doc, "1e"));              /* bad exponent     */
  CHECK(!parse_ok(&doc, "01"));              /* leading zero     */
  CHECK(!parse_ok(&doc, ""));                /* empty            */
  CHECK(!parse_ok(&doc, "nul"));             /* bad literal      */
}

/* Depth and node limits. */
{
  pai_json_doc_t doc;
  char deep[80];
  uint32_t i;
  /* Nest 33 arrays: one past PAI_JSON_MAX_DEPTH. */
  for (i = 0; i < 33; i++) {
    deep[i] = '[';
  }
  deep[33] = '0';
  for (i = 0; i < 33; i++) {
    deep[34 + i] = ']';
  }
  deep[67] = '\0';
  CHECK(!parse_ok(&doc, deep));
  CHECK(parse_ok(&doc, "[]"));
  pai_json_destroy(&doc);

  /* Node cap: an array of 9000 zeros must fail. */
  {
    char big[18002];
    uint32_t k = 0;
    big[k++] = '[';
    for (i = 0; i < 9000; i++) {
      if (i != 0) {
        big[k++] = ',';
      }
      big[k++] = '0';
    }
    big[k++] = ']';
    big[k] = '\0';
    CHECK(!parse_ok(&doc, big));
    pai_json_destroy(&doc);
  }
}

TEST_MAIN_END()

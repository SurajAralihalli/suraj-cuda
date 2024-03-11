#include <cuda_runtime.h>

template <int max_json_nesting_depth = curr_max_json_nesting_depth>
class json_generator {
 public:
  CUDF_HOST_DEVICE json_generator(char* _output, size_t _output_len)
    : output(_output), output_len(_output_len)
  {
  }
  CUDF_HOST_DEVICE json_generator() : output(nullptr), output_len(0) {}

  // create a nested child generator based on this parent generator
  // child generator is a view
  CUDF_HOST_DEVICE json_generator new_child_generator()
  {
    if (nullptr == output) {
      return json_generator();
    } else {
      return json_generator(output + output_len, 0);
    }
  }

  CUDF_HOST_DEVICE json_generator finish_child_generator(json_generator const& child_generator)
  {
    // logically delete child generator
    output_len += child_generator.get_output_len();
  }

  CUDF_HOST_DEVICE void write_start_array()
  {
    // TODO
  }

  CUDF_HOST_DEVICE void write_end_array()
  {
    // TODO
  }

  CUDF_HOST_DEVICE void copy_current_structure(json_parser<max_json_nesting_depth>& parser)
  {
    // 
    auto t = p.next_token();
    while(t!=empty) {
        if( t == START_OBJECT ) {
            
        }
    }
    
  }

  /**
   * Get current text from JSON parser and then write the text
   * Note: Because JSON strings contains '\' to do escape,
   * JSON parser should do unescape to remove '\' and JSON parser
   * then can not return a pointer and length pair (char *, len),
   * For number token, JSON parser can return a pair (char *, len)
   */
  CUDF_HOST_DEVICE void write_raw(json_parser<max_json_nesting_depth>& parser)
  {
    if (output) {
      auto copied = parser.try_copy_raw_text(output + output_len);
      output_len += copied;
    }
  }

  CUDF_HOST_DEVICE inline size_t get_output_len() const { return output_len; }

 private:
  char const* const output;
  size_t output_len;
  bool stack[max_json_nesting_depth];
  int index[max_json_nesting_depth];

};
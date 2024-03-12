#include <bits/stdc++.h>
#include <optional>
using namespace std;

enum class json_token {
    // start token
    INIT = 0,

    // successfully parsed the whole JSON string
    SUCCESS,

    // get error when parsing JSON string
    ERROR,

    // '{'
    START_OBJECT,

    // '}'
    END_OBJECT,

    // '['
    START_ARRAY,

    // ']'
    END_ARRAY,

    // e.g.: key1 in {"key1" : "value1"}
    FIELD_NAME,

    // e.g.: value1 in {"key1" : "value1"}
    VALUE_STRING,

    // e.g.: 123 in {"key1" : 123}
    VALUE_NUMBER_INT,

    // e.g.: 1.25 in {"key1" : 1.25}
    VALUE_NUMBER_FLOAT,

    // e.g.: true in {"key1" : true}
    VALUE_TRUE,

    // e.g.: false in {"key1" : false}
    VALUE_FALSE,

    // e.g.: null in {"key1" : null}
    VALUE_NULL
};


enum class item { OBJECT, ARRAY, EMPTY };

template <int max_json_nesting_depth>
class json_generator {

public:
    json_generator(char* _output, size_t _output_len): output(_output), output_len(_output_len) 
    {
    }

    json_generator(): output(nullptr), output_len(0) {}

    json_generator new_child_generator()
    {
        if (nullptr == output) {
            return json_generator();
        } else {
            return json_generator(output + output_len, 0);
        }
    }

    json_generator finish_child_generator(json_generator const& child_generator)
    {
        // logically delete child generator ?? TODO
        output_len += child_generator.get_output_len();
    }

    size_t get_output_len() const { return output_len; }

    char* get_output_start_position() const {
        return output;
    }

    char* get_current_output_position() const {
        return output + get_output_len();
    }

    void write_output(const char* str, size_t len) {
        if (output != nullptr) {
            std::memcpy(output + output_len, str, len);
            output_len = output_len + len;
        }
    }

    void write_start_array() {
        // new current
        initialize_new_context(item::ARRAY);
        // add [
        add_start_array();
    }

    void write_end_array() {
        // add ]
        add_end_array();
        // pop
        pop_curr_context();
    }

    void write_raw_value() {
        //check if current is a list and add comma if count[current] > 0
        if(!is_context_stack_empty() && is_array_context() && !is_first_member()) {
                add_comma();
        }
        //increment count
        register_member();
        //TODO
    }

    void copy_current_structure(list<json_token> parser) {
        while(!parser.empty()) {
            auto token = parser.front();
            // cout << enumToString(token) << endl;
            parser.pop_front();
            consume_token(token);
        }
    }

    std::string enumToString(json_token token) {
        switch(token) {
        case json_token::INIT: return "INIT";
        case json_token::SUCCESS: return "SUCCESS";
        case json_token::ERROR: return "ERROR";
        case json_token::START_OBJECT: return "START_OBJECT";
        case json_token::END_OBJECT: return "END_OBJECT";
        case json_token::START_ARRAY: return "START_ARRAY";
        case json_token::END_ARRAY: return "END_ARRAY";
        case json_token::FIELD_NAME: return "FIELD_NAME";
        case json_token::VALUE_STRING: return "VALUE_STRING";
        case json_token::VALUE_NUMBER_INT: return "VALUE_NUMBER_INT";
        case json_token::VALUE_NUMBER_FLOAT: return "VALUE_NUMBER_FLOAT";
        case json_token::VALUE_TRUE: return "VALUE_TRUE";
        case json_token::VALUE_FALSE: return "VALUE_FALSE";
        case json_token::VALUE_NULL: return "VALUE_NULL";
        }
        return "UNKNOWN_TOKEN";
    }

    void consume_token(json_token next_token) {

        if(next_token == json_token::INIT || next_token == json_token::SUCCESS) {
            return;
        }

        // check if value token
        if(next_token != json_token::FIELD_NAME && next_token !=json_token::END_OBJECT && next_token !=json_token::END_ARRAY) {
            consume_value_token(next_token);
            return;
        }

        // if object context
        if(!is_context_stack_empty())
        {
            //OBJECT Context
            if(is_object_context()) {
                if(next_token == json_token::FIELD_NAME) {
                    // add comma if count[current] > 0
                    if(!is_first_member()) {
                        add_comma();
                    }
                    // add key
                    write_output("\"key\"", 5);
                    // add :
                    write_output(":", 1);
                }
                else if(next_token == json_token::END_OBJECT) {
                    // add }
                    add_end_object();
                    // pop
                    pop_curr_context();
                }
            }
            //ARRAY Context
            else {
                if(next_token == json_token::END_ARRAY) {
                    // add ]
                    add_end_array();
                    // pop
                    pop_curr_context();
                }
            }
        }
        else {
            // Invalid JSON PARSER
            cout << "ERROR: INVALID JSON" << endl;
        }
        return;
    }

    void consume_value_token(json_token next_token) {

        //check if current is a list and add comma if count[current] > 0
        if(!is_context_stack_empty() && is_array_context() && !is_first_member()) {
                add_comma();
        }
        //increment count
        register_member();

        // if value token
        switch (next_token) {
            case json_token::START_OBJECT:
                // new current
                initialize_new_context(item::OBJECT);
                // add {
                add_start_object();
                break;
            case json_token::START_ARRAY:
                // new current
                initialize_new_context(item::ARRAY);
                // add [
                add_start_array();
                break;
            case json_token::VALUE_TRUE:
                // add true
                add_true();
                break;
            case json_token::VALUE_FALSE:
                // add false
                add_false();
                break;
            case json_token::VALUE_NULL:
                // add null
                add_null();
                break;
            case json_token::VALUE_NUMBER_INT:
                // add int
                write_output("1",1);
                break;
            case json_token::VALUE_NUMBER_FLOAT:
                // add float
                write_output("1.0",3);
                break;
            case json_token::VALUE_STRING:
                // add value
                write_output("\"value\"",7);
                break;
            default:
                // Invalid JSON PARSER
                cout << "ERROR: INVALID JSON" << endl;
                break;
        }
    }

    std::string getJsonString() const {
        if (output_len!=0) {
            return std::string(output, output_len);
        } else {
            return "";
        }
    }

    int getCurrent() const{
        return current;
    }

    bool is_complete_json() const {
        return getCurrent() == -1;
    }

private:
    bool has_members[max_json_nesting_depth] = {false};
    item type[max_json_nesting_depth] = {item::EMPTY};
    int current = -1;

    // char const* const output;
    char* const output;
    size_t output_len;

    bool is_context_stack_empty() {
        return current == -1;
    }

    bool is_object_context() {
        return type[current] == item::OBJECT;
    }

    bool is_array_context() {
        return type[current] == item::ARRAY;
    }

    void pop_curr_context() {
        has_members[current] = false;
        type[current] = item::EMPTY;
        current--;
    }

    bool is_first_member() {
        return has_members[current] == false;
    }

    void register_member() {
        has_members[current] = true;
    }

    void initialize_new_context(item _item) {
        current++;
        type[current] = _item;
        has_members[current] = false;
    }

    void add_start_array() {
        write_output("[",1);
    }

    void add_end_array() {
        write_output("]",1);
    }

    void add_start_object() {
        write_output("{",1);
    }

    void add_end_object() {
        write_output("}",1);
    }

    void add_true() {
        write_output("true",4);
    }

    void add_false() {
        write_output("false",4);
    }

    void add_null() {
        write_output("null",4);
    }

    void add_comma() {
        write_output(",", 1);
    }
};

int main() {
    // Example tokens
    list<json_token> parser = {
        json_token::START_OBJECT,
        json_token::FIELD_NAME,
        json_token::VALUE_STRING,
        json_token::FIELD_NAME,
        json_token::VALUE_STRING,
        json_token::FIELD_NAME,
        json_token::START_OBJECT,
        json_token::FIELD_NAME,
        json_token::VALUE_STRING,
        json_token::FIELD_NAME,
        json_token::START_OBJECT,
        json_token::FIELD_NAME,
        json_token::VALUE_STRING,
        json_token::FIELD_NAME,
        json_token::VALUE_STRING,
        json_token::END_OBJECT,
        json_token::FIELD_NAME,
        json_token::START_ARRAY,
        json_token::VALUE_STRING,
        json_token::VALUE_STRING,
        json_token::END_ARRAY,
        json_token::END_OBJECT,
        json_token::FIELD_NAME,
        json_token::START_ARRAY,
        json_token::START_OBJECT,
        json_token::FIELD_NAME,
        json_token::VALUE_STRING,
        json_token::FIELD_NAME,
        json_token::VALUE_STRING,
        json_token::END_OBJECT,
        json_token::START_OBJECT,
        json_token::FIELD_NAME,
        json_token::START_ARRAY,
        json_token::VALUE_STRING,
        json_token::VALUE_STRING,
        json_token::END_ARRAY,
        json_token::FIELD_NAME,
        json_token::START_OBJECT,
        json_token::FIELD_NAME,
        json_token::VALUE_STRING,
        json_token::FIELD_NAME,
        json_token::VALUE_STRING,
        json_token::END_OBJECT,
        json_token::END_OBJECT,
        json_token::END_ARRAY,
        json_token::END_OBJECT
    };

    list<json_token> parser2 = {
        json_token::END_ARRAY
    };

    const int max_output_len = 1000; // Derived after first pass
    char* const output = new char[max_output_len];

    const int curr_max_json_nesting_depth = 7;
    json_generator<curr_max_json_nesting_depth> generator(output, 0);
    generator.copy_current_structure(parser);

    // Get generated JSON string
    cout << "IS COMPLETE JSON: " << generator.is_complete_json() << endl;
    std::string jsonString = generator.getJsonString();
    std::cout << "Generated JSON: " << jsonString << std::endl;


    // INPUT JSON
    // {
    // "key1": "value1",
    // "key2": "value2",
    // "key3": {
    //     "key4": "value4",
    //     "key5": {
    //     "key6": "value6",
    //     "key7": "value7"
    //     },
    //     "key8": ["value8", "value9"]
    // },
    // "key10": [
    //     {
    //     "key11": "value11",
    //     "key12": "value12"
    //     },
    //     {
    //     "key13": ["value13", "value14"],
    //     "key15": {
    //         "key16": "value16",
    //         "key17": "value17"
    //     }
    //     }
    // ]
    // }


    // GENERATED JSON
    // {
    // "key": "value",
    // "key": "value",
    // "key": {
    //     "key": "value",
    //     "key": {
    //         "key": "value",
    //         "key": "value"
    //     },
    //     "key": [
    //         "value",
    //         "value"
    //     ]
    // },
    // "key": [
    //     {
    //         "key": "value",
    //         "key": "value"
    //     },
    //     {
    //         "key": [
    //             "value",
    //             "value"
    //         ],
    //         "key": {
    //             "key": "value",
    //             "key": "value"
    //         }
    //     }
    //     ]
    // }

    return 0;
}

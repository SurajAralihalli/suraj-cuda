#include <bits/stdc++.h>
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

template <int max_json_nesting_depth>
class JsonGenerator {
private:
    enum class item { OBJECT, ARRAY, EMPTY };
    std::ostringstream json;
    // int max_json_nesting_depth = 7; // Corrected declaration and initialization
    int count[max_json_nesting_depth] = {0};
    item type[max_json_nesting_depth] = {item::EMPTY};
    int current = -1;

public:
    JsonGenerator() {
    }

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
        count[current] = 0;
        type[current] = item::EMPTY;
        current--;
    }

    bool has_siblings() {
        return count[current] > 0;
    }

    void increment_siblings_count() {
        count[current]++;
    }

    void initialize_new_context(item _item) {
        current++;
        type[current] = _item;
        count[current] = 0;
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

        if(next_token == json_token::INIT) {
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
                    if(has_siblings()) {
                        json << ",";
                    }
                    // add key
                    json << "\"key\"";
                    // add :
                    json << ":";
                }
                else if(next_token == json_token::END_OBJECT) {
                    // add }
                    json << "}";
                    // pop
                    pop_curr_context();
                }
            }
            //ARRAY Context
            else {
                if(next_token == json_token::END_ARRAY) {
                    // add ]
                    json << "]";
                    // pop
                    pop_curr_context();
                }
            }
        }
        else {
            // Invalid JSON PARSER
        }
        
    }

    void consume_value_token(json_token next_token) {

        //check if current is a list and add comma if count[current] > 0
        if(!is_context_stack_empty() && is_array_context() && has_siblings()) {
                json << ",";
        }
        //increment count
        increment_siblings_count();

        // if value token
        switch (next_token) {
            case json_token::START_OBJECT:
                // new current
                initialize_new_context(item::OBJECT);
                // add {
                json << "{";
                break;
            case json_token::START_ARRAY:
                // new current
                initialize_new_context(item::ARRAY);
                // add [
                json << "[";
                break;
            case json_token::VALUE_TRUE:
                // add true
                json << "true";
                break;
            case json_token::VALUE_NUMBER_INT:
                // add int
                json << "1";
                break;
            case json_token::VALUE_NUMBER_FLOAT:
                // add float
                json << "1.0";
                break;
            case json_token::VALUE_FALSE:
                // add false
                json << "false";
                break;
            case json_token::VALUE_STRING:
                // add value
                json << "\"value\"";
                break;
            default:
                // Invalid JSON PARSER
                break;
        }

    }

    void copyCurrentStructure(list<json_token> parser) {
        while(!parser.empty()) {
            auto token = parser.front();
            // cout << enumToString(token) << endl;
            parser.pop_front();
            consume_token(token);
        }
    }

    std::string getJsonString() const {
        return json.str();
    }

    int getCurrent() {
        return current;
    }
};

int main() {
    // Example tokens (for demonstration purposes)
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

    // list<json_token> parser = {
    //     json_token::VALUE_STRING
    // };

    const int curr_max_json_nesting_depth = 7;
    JsonGenerator<curr_max_json_nesting_depth> generator;
    generator.copyCurrentStructure(parser);

    // Get generated JSON string
    cout << "IS VALID JSON PARSER: " << (generator.getCurrent() == -1) << endl;
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

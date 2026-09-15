#pragma once

#include <string>
#include "llama-model.h"

struct parameter_offloader_model_i {
    bool (*weight_supported)(const std::string & name);
    void (*configure_dense_read_ops)(bool * dense_read_ops);
};

extern parameter_offloader_model_i parameter_offloader_deepseek2_i;
extern parameter_offloader_model_i parameter_offloader_gpt_oss_i;
extern parameter_offloader_model_i parameter_offloader_deepseek4_i;

parameter_offloader_model_i * parameter_offloader_get_model_i(llama_model  * model);
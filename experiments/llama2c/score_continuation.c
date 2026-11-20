#include <stdint.h>

#define TESTING
#include "run.c"

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: score_continuation MODEL TOKENIZER PROMPT CONTINUATION\n");
        return EXIT_FAILURE;
    }

    Transformer model;
    Tokenizer tokenizer;
    build_transformer(&model, argv[1]);
    build_tokenizer(&tokenizer, argv[2], model.config.vocab_size);

    const size_t prompt_bytes = strlen(argv[3]);
    const size_t total_bytes = prompt_bytes + strlen(argv[4]);
    char *text = malloc(total_bytes + 1);
    int *prefix = malloc((prompt_bytes + 3) * sizeof(*prefix));
    int *tokens = malloc((total_bytes + 3) * sizeof(*tokens));
    if (!text || !prefix || !tokens) return EXIT_FAILURE;
    memcpy(text, argv[3], prompt_bytes);
    strcpy(text + prompt_bytes, argv[4]);

    int prefix_count = 0;
    int token_count = 0;
    encode(&tokenizer, argv[3], 1, 0, prefix, &prefix_count);
    encode(&tokenizer, text, 1, 0, tokens, &token_count);

    int status = EXIT_FAILURE;
    if (token_count > model.config.seq_len || prefix_count >= token_count ||
        memcmp(prefix, tokens, prefix_count * sizeof(*prefix)) != 0) {
        fprintf(stderr, "The candidate exceeded the context limit or changed the prompt tokenization.\n");
    } else {
        double loss = 0.0;
        for (int pos = 0; pos < token_count - 1; ++pos) {
            float *logits = forward(&model, tokens[pos], pos);
            if (pos + 1 < prefix_count) continue;
            softmax(logits, model.config.vocab_size);
            loss -= log(fmax(logits[tokens[pos + 1]], 1e-30));
        }
        printf("%.8f\n", loss / (token_count - prefix_count));
        status = EXIT_SUCCESS;
    }

    free(tokens);
    free(prefix);
    free(text);
    free_tokenizer(&tokenizer);
    free_transformer(&model);
    return status;
}

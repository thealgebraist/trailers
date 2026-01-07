#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

std::vector<int64_t> read_tokens_txt(const std::string &path) {
    std::ifstream f(path);
    std::vector<int64_t> tokens;
    if (!f) return tokens;
    int64_t v;
    while (f >> v) tokens.push_back(v);
    return tokens;
}

int main() {
    std::vector<std::string> files = {"bark_token_ids.txt", "flux_token_ids.txt", "ldm2_token_ids.txt"};
    for (auto &fn : files) {
        auto tokens = read_tokens_txt(fn);
        std::cout << fn << ": ";
        if (tokens.empty()) {
            std::cout << "(missing or empty)\n";
            continue;
        }
        std::cout << tokens.size() << " tokens; first 10: ";
        for (size_t i=0;i<std::min<size_t>(tokens.size(),10);++i) std::cout << tokens[i] << (i+1<10?" ":"\n");
    }
    return 0;
}

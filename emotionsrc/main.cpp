#include <iostream>
#include "FHEController.h"
#include <chrono>

#define GREEN_TEXT "\033[1;32m"
#define RED_TEXT "\033[1;31m"

using namespace std::chrono;

enum class Parameters { Generate, Load };

void setup_environment(int argc, char *argv[]);

FHEController controller;

vector<Ctxt> encoder1();
Ctxt encoder2(vector<Ctxt> input);
Ctxt pooler(Ctxt input);
Ctxt classifier(Ctxt input);

//Set to True to test the program on the IDE
bool IDE_MODE = false;

string input_folder;

//Argument
string text;

//<OPTIONS>
bool verbose = false;
bool security128bits = false;
Parameters p = Parameters::Load;
bool plain;

int main(int argc, char *argv[]) {
    setup_environment(argc, argv);

    if (p == Parameters::Generate) {
        system("mkdir -p ../keys");
        controller.generate_context(true, security128bits);
        vector<int> rotations = {1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 684, 1024, 2048, 4096, 8192, -1, -2, -3, -4, -5, -8, -16, -32, -64};
        controller.generate_bootstrapping_and_rotation_keys(rotations, 16384, true, "rotation_keys.txt");
        return 0;
    } else if (p == Parameters::Load) {
        controller.load_context(false);
        controller.load_bootstrapping_and_rotation_keys("rotation_keys.txt", 16384, false);
    }

    system("mkdir -p ../checkpoint");

    if (verbose) cout << "\nSERVER-SIDE\nThe evaluation of the circuit started." << endl;

    auto start = high_resolution_clock::now();

    if (input_folder.empty()) {
        cerr << "The input folder \"" << input_folder << "\" is empty!";
        exit(1);
    }

    vector<Ctxt> encoder1output;
    Ctxt encoder2output;

    encoder1output = encoder1();
    encoder1output = controller.load_vector("../checkpoint/encoder1output.bin");

    encoder2output = encoder2(encoder1output);
    encoder2output = controller.load_ciphertext("../checkpoint/encoder2output.bin");

    Ctxt pooled = pooler(encoder2output);
    pooled = controller.load_ciphertext("../checkpoint/pooled.bin");

    Ctxt classified = classifier(pooled);

    if (verbose) cout << "The circuit has been evaluated, the results are sent back to the client" << endl << endl;
    if (verbose) cout << "CLIENT-SIDE" << endl;

    /*
    if (verbose)
        controller.print(classified, 2, "Output logits");

    vector<double> plain_result = controller.decrypt_tovector(classified, 2);
    */
    
    vector<double> plain_result = controller.decrypt_tovector6(classified, controller.num_slots);
    cout << "plain_result: [" 
    << plain_result[0] << ", " << plain_result[1] << ", " << plain_result[2] << ", " 
    << plain_result[3] << ", " << plain_result[4] << ", " << plain_result[5] << "]" << endl;

    int timing = (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0;
    if (verbose) cout << endl << "The evaluation of the FHE circuit took: " << timing << " seconds." << endl;
    cout << endl << "The evaluation of the FHE circuit took: " << timing << " seconds." << endl;

    vector<string> emotions = {"sadness", "joy", "love", "anger", "fear", "surprise"};
    int max_index = std::max_element(plain_result.begin(), plain_result.end()) - plain_result.begin();

    if (plain) {
        cout << "Outcomes:" << endl << "FHE              : ";
        cout << emotions[max_index] << endl;
        system(("python3 ../newsrc/python/PlainCircuit.py \"" + text + "\"").c_str());
        
        // Construct the result string for all emotions
        string result_str = "[";
        for (int i = 0; i < 6; i++) {
            result_str += to_string(plain_result[i]);
            if (i < 5) result_str += ", ";
        }
        result_str += "]";
        
        system(("python3 ../newsrc/python/Precision.py \"" + text + "\" " + "\"" + result_str + "\" " + to_string(timing)).c_str());
    } else {
        cout << "Outcome: " << emotions[max_index] << endl;
    }

}

Ctxt classifier(Ctxt input) {
    Ptxt weight = controller.read_plain_input("../weights-emotion/classifier_weight.txt", input->GetLevel());
    Ptxt bias = controller.read_plain_expanded_input("../weights-emotion/classifier_bias.txt", input->GetLevel());

    Ctxt output = controller.mult(input, weight);

    output = controller.rotsum(output, 128, 1);

    output = controller.add(output, bias);

    vector<double> mask;
    for (int i = 0; i < controller.num_slots; i++) {
        mask.push_back(0);
    }

    mask[0] = 1;
    mask[128] = 1;
    mask[256] = 1;
    mask[384] = 1;
    mask[512] = 1;
    mask[640] = 1;

    output = controller.mult(output, controller.encrypt(mask, output->GetLevel()));

    // return the output directly
    // output = controller.add(output, controller.rotate(controller.rotate(output, -1), 128));

    cout << "classifier level: " << output -> GetLevel() << endl;
    return output;
}
Ctxt pooler(Ctxt input) {
    auto start = high_resolution_clock::now();

    double tanhScale = 1 / 28.66;

    Ptxt weight = controller.read_plain_input("../weights-emotion/pooler_dense_weight.txt", input->GetLevel(), tanhScale);
    Ptxt bias = controller.read_plain_repeated_input("../weights-emotion/pooler_dense_bias.txt", input->GetLevel(), tanhScale);

    Ctxt output = controller.mult(input, weight);

    output = controller.rotsum(output, 128, 128);

    output = controller.add(output, bias);

    //cout << "level: " << output -> GetLevel() << endl;
    output = controller.bootstrap(output);
    //cout << "level: " << output -> GetLevel() << endl;

    output = controller.eval_tanh_function(output, -1, 1, tanhScale, 132);

    cout << "pooler level: " << output -> GetLevel() << endl;
    if (verbose) cout << "The evaluation of Pooler took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    if (verbose) controller.print(output, 128, "Pooler (Repeated)");

    controller.save(output, "../checkpoint/pooled.bin");

    return output;
}
Ctxt encoder2(vector<Ctxt> inputs) {
    auto start = high_resolution_clock::now();

    Ptxt query_w = controller.read_plain_input("../weights-emotion/layer1_attself_query_weight.txt", inputs[0]->GetLevel());
    Ptxt query_b = controller.read_plain_repeated_input("../weights-emotion/layer1_attself_query_bias.txt", inputs[0]->GetLevel());
    Ptxt key_w = controller.read_plain_input("../weights-emotion/layer1_attself_key_weight.txt", inputs[0]->GetLevel());
    Ptxt key_b = controller.read_plain_repeated_input("../weights-emotion/layer1_attself_key_bias.txt", inputs[0]->GetLevel());

    vector<Ctxt> Q = controller.matmulRE(inputs, query_w, query_b);
    vector<Ctxt> K = controller.matmulRE(inputs, key_w, key_b);

    Ctxt K_wrapped = controller.wrapUpRepeated(K);

    Ctxt scores = controller.matmulScores(Q, K_wrapped);
    
    //cout << "level: " << scores -> GetLevel() << endl;
    scores = controller.bootstrap(scores);
    //cout << "level: " << scores -> GetLevel() << endl;

    scores = controller.eval_exp(scores, inputs.size());

    scores = controller.mult(scores, 1 / 5000.0); //Here values are scaled down in order to achieve better accuracy with bootstrapping

    //cout << "level: " << scores -> GetLevel() << endl;
    scores = controller.bootstrap(scores);
    //cout << "level: " << scores -> GetLevel() << endl;

    scores = controller.mult(scores, 5000.0);
    //cout << "after mult level: " << scores -> GetLevel() << endl;
    
    Ctxt scores_sum = controller.rotsum(scores, 128, 128);
    //cout << "after rotsum level: " << scores_sum -> GetLevel() << endl;

    controller.print_min_max(scores_sum);

    // cost 10 circuit level, but level after rotsum is 15 so we have to bootstrap after the rotsum
    Ctxt scores_denominator = controller.eval_inverse_naive_2(scores_sum, 1.31, 68336.18, 1); // 38018.86 // 16542.94 // 68336.18 // 6514.89
    
    // check
    //cout << "level: " << scores_denominator -> GetLevel() << endl;
    scores_denominator = controller.bootstrap(scores_denominator);
    //cout << "level: " << scores_denominator -> GetLevel() << endl;

    scores = controller.mult(scores, scores_denominator);

    vector<Ctxt> unwrapped_scores = controller.unwrapScoresExpanded(scores, inputs.size());

    Ptxt value_w = controller.read_plain_input("../weights-emotion/layer1_attself_value_weight.txt", inputs[0]->GetLevel());
    Ptxt value_b = controller.read_plain_repeated_input("../weights-emotion/layer1_attself_value_bias.txt", inputs[0]->GetLevel());
    
    vector<Ctxt> V = controller.matmulRE(inputs, value_w, value_b);

    Ctxt V_wrapped = controller.wrapUpRepeated(V);
    
    vector<Ctxt> output = controller.matmulRE(unwrapped_scores, V_wrapped, 128, 128);

    // level 26
    cout << "self-attention level: " << output[0] -> GetLevel() << endl;
    if (verbose) cout << "The evaluation of Self-Attention took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    if (verbose) controller.print(output[0], 128, "Self-Attention (Repeated)");
    //Qua la precisione è 0.9868

    /*
     * I remove all the ciphertexts except the first corresponding to the CLS token
     */
    Ctxt copyFirst = output[0];
    output.clear();
    output.push_back(copyFirst);

    start = high_resolution_clock::now();

    Ptxt dense_w = controller.read_plain_input("../weights-emotion/layer1_selfoutput_weight.txt", output[0]->GetLevel());
    Ptxt dense_b = controller.read_plain_expanded_input("../weights-emotion/layer1_selfoutput_bias.txt", output[0]->GetLevel() + 1); //Bias fai solo 12 ripetiz

    output = controller.matmulCR(output, dense_w, dense_b);

    for (int i = 0; i < output.size(); i++) {
        output[i] = controller.add(output[i], inputs[i]);
    }

    Ctxt wrappedOutput = controller.wrapUpExpanded(output);

    Ptxt precomputed_mean = controller.read_plain_repeated_input("../emotion-precompute/layer1_selfoutput_mean.txt", wrappedOutput->GetLevel(), -1);
    wrappedOutput = controller.add(wrappedOutput, precomputed_mean);

    //cout << "level: " << wrappedOutput -> GetLevel() << endl;
    wrappedOutput = controller.bootstrap(wrappedOutput);
    //cout << "level: " << wrappedOutput -> GetLevel() << endl;

    Ptxt vy = controller.read_plain_input("../emotion-precompute/layer1_selfoutput_vy.txt", wrappedOutput->GetLevel(), 1);
    wrappedOutput = controller.mult(wrappedOutput, vy);
    Ptxt bias = controller.read_plain_expanded_input("../emotion-precompute/layer1_selfoutput_normbias.txt", wrappedOutput->GetLevel(), 1, inputs.size());
    wrappedOutput = controller.add(wrappedOutput, bias);

    Ctxt output_copy = wrappedOutput->Clone(); //Required at the last layernorm

    output = controller.unwrapExpanded(wrappedOutput, inputs.size());

    cout << "self-output level: " << output[0] -> GetLevel() << endl;
    if (verbose) cout << "The evaluation of Self-Output took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    if (verbose) controller.print_expanded(output[0], 0, 128, "Self-Output (Expanded)");
    //Fino a qui ottengo precisione 0.9828


    start = high_resolution_clock::now();

    double GELU_max_abs_value = 1 / 14.7;

    Ptxt intermediate_w_1 = controller.read_plain_input("../weights-emotion/layer1_intermediate_weight1.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_2 = controller.read_plain_input("../weights-emotion/layer1_intermediate_weight2.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_3 = controller.read_plain_input("../weights-emotion/layer1_intermediate_weight3.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_4 = controller.read_plain_input("../weights-emotion/layer1_intermediate_weight4.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);

    vector<Ptxt> dense_weights = {intermediate_w_1, intermediate_w_2, intermediate_w_3, intermediate_w_4};

    Ptxt intermediate_bias = controller.read_plain_input("../weights-emotion/layer1_intermediate_bias.txt", output[0]->GetLevel() + 1, GELU_max_abs_value);

    output = controller.matmulRElarge(output, dense_weights, intermediate_bias);

    output = controller.generate_containers(output, nullptr);

    for (int i = 0; i < output.size(); i++) {
        output[i] = controller.eval_gelu_function(output[i], -1, 1, GELU_max_abs_value, 48); // 51

        //cout << "level: " << output[i] -> GetLevel() << endl;
        output[i] = controller.bootstrap(output[i]);
        //cout << "level: " << output[i] -> GetLevel() << endl;
    }

    vector<vector<Ctxt>> unwrappedLargeOutput = controller.unwrapRepeatedLarge(output, output.size());

    cout << "intermediate level: " << unwrappedLargeOutput[0][0] -> GetLevel() << endl;
    if (verbose) cout << "The evaluation of Intermediate took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    if (verbose) controller.print(unwrappedLargeOutput[0][0], 128, "Intermediate (Containers)");

    Ptxt output_w_1 = controller.read_plain_input("../weights-emotion/layer1_output_weight1.txt", output[0]->GetLevel());
    Ptxt output_w_2 = controller.read_plain_input("../weights-emotion/layer1_output_weight2.txt", output[0]->GetLevel());
    Ptxt output_w_3 = controller.read_plain_input("../weights-emotion/layer1_output_weight3.txt", output[0]->GetLevel());
    Ptxt output_w_4 = controller.read_plain_input("../weights-emotion/layer1_output_weight4.txt", output[0]->GetLevel());

    Ptxt output_bias = controller.read_plain_expanded_input("../weights-emotion/layer1_output_bias.txt", output[0]->GetLevel() + 1);

    output = controller.matmulCRlarge(unwrappedLargeOutput, {output_w_1, output_w_2, output_w_3, output_w_4}, output_bias);
    wrappedOutput = controller.wrapUpExpanded(output);

    wrappedOutput = controller.add(wrappedOutput, output_copy);

    precomputed_mean = controller.read_plain_repeated_input("../emotion-precompute/layer1_output_mean.txt", wrappedOutput->GetLevel(), -1);
    wrappedOutput = controller.add(wrappedOutput, precomputed_mean);

    vy = controller.read_plain_input("../emotion-precompute/layer1_output_vy.txt", wrappedOutput->GetLevel(), 1);
    wrappedOutput = controller.mult(wrappedOutput, vy);
    bias = controller.read_plain_expanded_input("../emotion-precompute/layer1_output_normbias.txt", wrappedOutput->GetLevel(), 1, inputs.size());
    wrappedOutput = controller.add(wrappedOutput, bias);

    output = controller.unwrapExpanded(wrappedOutput, inputs.size());

    cout << "output level: " << output[0] -> GetLevel() << endl;
    if (verbose) cout << "The evaluation of Output took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    if (verbose) controller.print_expanded(output[0], 0, 128, "Output (Expanded)");

    controller.save(output[0], "../checkpoint/encoder2output.bin");

    return output[0];
}
vector<Ctxt> encoder1() {
    auto start = high_resolution_clock::now();

    int inputs_count = 0;

    std::filesystem::path p1 { input_folder };

    for (__attribute__((unused)) auto& p : std::filesystem::directory_iterator(p1))
    {
        ++inputs_count;
    }

    if (verbose) cout << inputs_count << " inputs found!" << endl << endl;

    vector<Ctxt> inputs;
    for (int i = 0; i < inputs_count; i++) {
        inputs.push_back(controller.read_expanded_input(input_folder + "input_" + to_string(i) + ".txt"));
    }

    Ptxt query_w = controller.read_plain_input("../weights-emotion/layer0_attself_query_weight.txt");
    Ptxt query_b = controller.read_plain_repeated_input("../weights-emotion/layer0_attself_query_bias.txt");
    Ptxt key_w = controller.read_plain_input("../weights-emotion/layer0_attself_key_weight.txt");
    Ptxt key_b = controller.read_plain_repeated_input("../weights-emotion/layer0_attself_key_bias.txt");

    vector<Ctxt> Q = controller.matmulRE(inputs, query_w, query_b);
    vector<Ctxt> K = controller.matmulRE(inputs, key_w, key_b);

    Ctxt K_wrapped = controller.wrapUpRepeated(K);

    Ctxt scores = controller.matmulScores(Q, K_wrapped);
    scores = controller.eval_exp(scores, inputs.size());

    Ctxt scores_sum = controller.rotsum(scores, 128, 128);
    
    //cout << "level: " << scores_sum -> GetLevel() << endl;
    Ctxt scores_denominator = controller.eval_inverse_naive(scores_sum, 1.63, 25030.75); // 25030.75 // 8973.86 // 4670.44
    //cout << "level: " << scores_denominator -> GetLevel() << endl;

    // cout << "scores level: " << scores -> GetLevel() << endl;
    scores = controller.mult(scores, scores_denominator);
    //scores = controller.bootstrap(scores);
    //cout << "scores level: " << scores -> GetLevel() << endl;

    vector<Ctxt> unwrapped_scores = controller.unwrapScoresExpanded(scores, inputs.size());
    
    Ptxt value_w = controller.read_plain_input("../weights-emotion/layer0_attself_value_weight.txt", scores->GetLevel() - 2);
    Ptxt value_b = controller.read_plain_repeated_input("../weights-emotion/layer0_attself_value_bias.txt", scores->GetLevel() - 1);

    vector<Ctxt> V = controller.matmulRE(inputs, value_w, value_b);
    Ctxt V_wrapped = controller.wrapUpRepeated(V);

    vector<Ctxt> output = controller.matmulRE(unwrapped_scores, V_wrapped, 128, 128);

    cout << "self-attention level: " << output[0] -> GetLevel() << endl;
    if (verbose) cout << "The evaluation of Self-Attention took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    if (verbose) controller.print(output[0], 128, "Self-Attention (Repeated)");
    //Fino a qui ottengo precisione 0.9934

    start = high_resolution_clock::now();

    Ptxt dense_w = controller.read_plain_input("../weights-emotion/layer0_selfoutput_weight.txt", output[0]->GetLevel());
    Ptxt dense_b = controller.read_plain_expanded_input("../weights-emotion/layer0_selfoutput_bias.txt", output[0]->GetLevel() + 1); //Bias fai solo 12 ripetiz
    //cout << "level: " << dense_w -> GetLevel() << endl;
    output = controller.matmulCR(output, dense_w, dense_b);
    //cout << "level: " << output[0] -> GetLevel() << endl;

    for (int i = 0; i < output.size(); i++) {
        output[i] = controller.add(output[i], inputs[i]);
    }
    //cout << "loop level: " << output[0] -> GetLevel() << endl;

    Ctxt wrappedOutput = controller.wrapUpExpanded(output);
    //cout << "wrappedOutput level: " << wrappedOutput -> GetLevel() << endl;

    Ptxt precomputed_mean = controller.read_plain_repeated_input("../emotion-precompute/layer0_selfoutput_mean.txt", wrappedOutput->GetLevel(), -1);
    wrappedOutput = controller.add(wrappedOutput, precomputed_mean);
    //cout << "add level: " << wrappedOutput -> GetLevel() << endl;

    // since degree +1 depth, require bootstrapping here
    //cout << "level: " << wrappedOutput -> GetLevel() << endl;
    wrappedOutput = controller.bootstrap(wrappedOutput);
    //cout << "level: " << wrappedOutput -> GetLevel() << endl;
    Ptxt vy = controller.read_plain_input("../emotion-precompute/layer0_selfoutput_vy.txt", wrappedOutput->GetLevel(), 1);
    //cout << "level: " << wrappedOutput -> GetLevel() << endl;
    wrappedOutput = controller.mult(wrappedOutput, vy);
    //cout << "level: " << wrappedOutput -> GetLevel() << endl;

    Ptxt bias = controller.read_plain_expanded_input("../emotion-precompute/layer0_selfoutput_normbias.txt", wrappedOutput->GetLevel(), 1, inputs.size());
    wrappedOutput = controller.add(wrappedOutput, bias);
    //cout << "level: " << wrappedOutput -> GetLevel() << endl;

    // origin (move to the a upper line)
    //wrappedOutput = controller.bootstrap(wrappedOutput);
    
    Ctxt output_copy = wrappedOutput->Clone(); //Required at the last layernorm

    output = controller.unwrapExpanded(wrappedOutput, inputs.size());

    cout << "self-output level: " << output[0] -> GetLevel() << endl;
    if (verbose) cout << "The evaluation of Self-Output took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    if (verbose) controller.print_expanded(output[0], 0, 128, "Self-Output (Expanded)");
    //Fino a qui ottengo precisione 0.9964

    start = high_resolution_clock::now();

    double GELU_max_abs_value = 1 / 13.7;

    Ptxt intermediate_w_1 = controller.read_plain_input("../weights-emotion/layer0_intermediate_weight1.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_2 = controller.read_plain_input("../weights-emotion/layer0_intermediate_weight2.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_3 = controller.read_plain_input("../weights-emotion/layer0_intermediate_weight3.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_4 = controller.read_plain_input("../weights-emotion/layer0_intermediate_weight4.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);

    vector<Ptxt> dense_weights = {intermediate_w_1, intermediate_w_2, intermediate_w_3, intermediate_w_4};

    Ptxt intermediate_bias = controller.read_plain_input("../weights-emotion/layer0_intermediate_bias.txt", output[0]->GetLevel() + 1, GELU_max_abs_value);

    output = controller.matmulRElarge(output, dense_weights, intermediate_bias);

    output = controller.generate_containers(output, nullptr);

    for (int i = 0; i < output.size(); i++) {
        output[i] = controller.eval_gelu_function(output[i], -1, 1, GELU_max_abs_value, 38); // 41
        //cout << "level: " << output[i] -> GetLevel() << endl;
        output[i] = controller.bootstrap(output[i]);
        //cout << "level: " << output[i] -> GetLevel() << endl;
    }

    vector<vector<Ctxt>> unwrappedLargeOutput = controller.unwrapRepeatedLarge(output, inputs.size());

    cout << "intermediate level: " << unwrappedLargeOutput[0][0] -> GetLevel() << endl;
    if (verbose) cout << "The evaluation of Intermediate took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    if (verbose) controller.print(unwrappedLargeOutput[0][0], 128, "Intermediate (Containers)");
    //Fino a qui ottengo precisione 0.9957

    Ptxt output_w_1 = controller.read_plain_input("../weights-emotion/layer0_output_weight1.txt", unwrappedLargeOutput[0][0]->GetLevel());
    Ptxt output_w_2 = controller.read_plain_input("../weights-emotion/layer0_output_weight2.txt", unwrappedLargeOutput[0][0]->GetLevel());
    Ptxt output_w_3 = controller.read_plain_input("../weights-emotion/layer0_output_weight3.txt", unwrappedLargeOutput[0][0]->GetLevel());
    Ptxt output_w_4 = controller.read_plain_input("../weights-emotion/layer0_output_weight4.txt", unwrappedLargeOutput[0][0]->GetLevel());

    Ptxt output_bias = controller.read_plain_expanded_input("../weights-emotion/layer0_output_bias.txt", unwrappedLargeOutput[0][0]->GetLevel() + 1);

    output = controller.matmulCRlarge(unwrappedLargeOutput, {output_w_1, output_w_2, output_w_3, output_w_4}, output_bias);
    wrappedOutput = controller.wrapUpExpanded(output);

    wrappedOutput = controller.add(wrappedOutput, output_copy);

    precomputed_mean = controller.read_plain_repeated_input("../emotion-precompute/layer0_output_mean.txt", wrappedOutput->GetLevel(), -1);
    wrappedOutput = controller.add(wrappedOutput, precomputed_mean);

    vy = controller.read_plain_input("../emotion-precompute/layer0_output_vy.txt", wrappedOutput->GetLevel(), 1);
    wrappedOutput = controller.mult(wrappedOutput, vy);
    bias = controller.read_plain_expanded_input("../emotion-precompute/layer0_output_normbias.txt", wrappedOutput->GetLevel(), 1, inputs.size());
    wrappedOutput = controller.add(wrappedOutput, bias);

    output = controller.unwrapExpanded(wrappedOutput, inputs.size());

    cout << "output level: " << output[0] -> GetLevel() << endl;
    if (verbose) cout << "The evaluation of Output took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    if (verbose) controller.print_expanded(output[0], 0, 128,"Output (Expanded)");
    //Fino a qui ottengo precisione 0.9965

    controller.save(output, "../checkpoint/encoder1output.bin");

    return output;
}

void setup_environment(int argc, char *argv[]) {
    string command;

    if (IDE_MODE) {
        filesystem::remove_all("../emotionsrc/tmp_embeddings");
        system("mkdir ../emotionsrc/tmp_embeddings");

        input_folder = "../emotionsrc/tmp_embeddings/";

        text = "This is a bad movie.";
        cout << "\nCLIENT-SIDE\nTokenizing the following sentence: '" << text << "'" << endl;
        command = "python3 ../emotionsrc/python/ExtractEmbeddings.py \"" + text + "\"";

        system(command.c_str());

        verbose = true;
        return;
    }

    if (argc < 2) {
        cout << "This is FHEBERT-Tiny, an encrypted text classifier based on BERT-tiny. It relies on the CKKS homomorphic encryption scheme.\n\nUsage: ./FHEBERT-tiny <text_input> [OPTIONS]\n\nthe following [OPTIONS] are available:\n--verbose: activates verbose mode\n--secure: creates parameters with 128 bits of security. Use only if necessary, as it adds computational overhead \n\nExample:\n./FHEBERT-tiny \"I wonder if this text will be well classified!\" --verbose\n";
        exit(0);
    } else {
        if (string(argv[1]) == "--generate_keys")
        {
            if (argc > 2 && string(argv[2]) == "--secure") {
                security128bits = true;
            }

            p = Parameters::Generate;
            return;
        }

        text = argv[1];

        //Removing any previous embedding
        filesystem::remove_all("../emotionsrc/tmp_embeddings/");
        system("mkdir ../emotionsrc/tmp_embeddings");

        input_folder = "../emotionsrc/tmp_embeddings/";


        for (int i = 2; i < argc; i++) {
            if (string(argv[i]) == "--verbose") {
                verbose = true;
            }

            if (string(argv[i]) == "--plain") {
                plain = true;
            }
        }

        if (verbose) cout << "\nCLIENT-SIDE\nTokenizing the following sentence: '" << text << "'" << endl;
        command = "python3 ../emotionsrc/python/ExtractEmbeddings.py \"" + text + "\"";
        system(command.c_str());
    }

}

#include <iostream>
#include <string>
// for reading writing
#include <fstream>
//for exp and log  
#include <math.h>
//For random number generator
#include <cstdlib>
//#define double long double
#include <algorithm>

double** W1 = new double* [16];
double** W2 = new double* [10];

double sigmoid(double a) {
    return (1 / (1 + exp(-a)));
}

double LeakyRelu(double a) {
    return a >= 0 ? a : 0.01 * a;
}

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

double GenerateRandom() {
    double random_float = static_cast<double>(rand()) / (RAND_MAX);
    return random_float;
}

int DerivLeaky(double M) {
    if (M > 0)
        return 1;
    else
        return 0.01;
}

void saveArrayToFile(const char* filename, double** array, int rows, int cols) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            outFile << array[i][j] << ' ';
        }
        outFile << std::endl;
    }

    outFile.close();
}


void read_mnist_cv(const char* image_filename, const char* label_filename) {
    for (int i = 0; i < 16; i++) {
        W1[i] = new double[785];
    }

    for (int i = 0; i < 10; i++) {
        W2[i] = new double[17];
    }

    double** W1new = new double* [16];
    double** W2new = new double* [10];

    for (int i = 0; i < 16; i++) {
        W1new[i] = new double[785];
    }

    for (int i = 0; i < 10; i++) {
        W2new[i] = new double[17];
    }


    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 785; j++) {
            W1[i][j] = GenerateRandom();
        }
    }

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 17; j++) {
            W2[i][j] = GenerateRandom();
        }
    }

    


    
   
    
    // Open files

    std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);

    // Read the magic and the meta data
    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;
    

    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2051) {
        std::cout << "Incorrect image file magic: " << magic << std::endl;
        return;
    }

    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2049) {
        std::cout << "Incorrect image file magic: " << magic << std::endl;
        return;
    }

    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    if (num_items != num_labels) {
        std::cout << "image file nums should equal to label num" << std::endl;
        return;
    }

    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    //std::cout << "image and label num is: " << num_items << std::endl;
    //std::cout << "image rows: " << rows << ", cols: " << cols << std::endl;

    char label;
    char* pixels = new char[rows * cols];


    double* Hidden_Layer1 = new double[16];
    double* Hidden_Layer1new = new double[16];
    double* Output_Layer = new double[10];
    double* Output_Layer_new = new double[10];
    double* RealOutput = new double[10];

    
    


    // Print the pixels array as a 2D matrix
    /*
    * image_file.read(pixels, rows * cols);
    label_file.read(&label, 1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << static_cast<int>(static_cast<unsigned char>(pixels[i * cols + j])) << ' ';
        }
        std::cout << std::endl;
    }
    */
    double* dlBYdonew = new double[10];
    double** donewBYdo = new double* [10];
    double** dlBYdw2 = new double* [10];
    double* doBYdw2 = new double[17];
    double** dlBYdw1 = new double* [16];
    
    for (int i = 0; i < 10; i++) {
        donewBYdo[i] = new double[10];
    }
    for (int i = 0; i < 10; i++) {
        dlBYdw2[i] = new double[17];
    }
    for (int i = 0; i < 16; i++) {
        dlBYdw1[i] = new double[785];
    }

    bool f = true;

    int BatchSize = 300;
    for (int item_id = 0; item_id < num_items/BatchSize; item_id++) {
        for (int Batch_no = 0; Batch_no < BatchSize; Batch_no++) {



            image_file.read(pixels, rows * cols);
            label_file.read(&label, 1);

            double* tempdlBYdw1 = new double[10];
            double* tempdlBYdw2 = new double[16];
            std::fill(tempdlBYdw1, tempdlBYdw1 + 10, 0.0);
            std::fill(tempdlBYdw2, tempdlBYdw2 + 16, 0.0);

            // calculating Layer 1
            for (int i = 0; i < 16; i++) {
                Hidden_Layer1[i] = W1[i][0];
                for (int j = 1; j < 785; j++) {
                    int pixelValue = static_cast<int>(static_cast<unsigned char>(pixels[j-1]));
                    Hidden_Layer1[i] = Hidden_Layer1[i] + W1[i][j] * ((double)pixelValue / 256.0f);
                }
                Hidden_Layer1new[i] = LeakyRelu(Hidden_Layer1[i]);
            }
            
            // calculating Layer 2
            
            double SoftSum = 0;
            // calculating Output
            for (int i = 0; i < 10; i++) {
                Output_Layer[i] = W2[i][0];
                for (int j = 1; j < 17; j++) {
                    Output_Layer[i] = Output_Layer[i] + W2[i][j] * Hidden_Layer1new[j - 1];
                }
                SoftSum = SoftSum + exp(Output_Layer[i]);
            }
            

            for (int i = 0; i < 10; i++) {
                Output_Layer_new[i] = exp(Output_Layer[i]) / SoftSum;
                //Output_Layer_new[i] = Output_Layer_new[i] >= 1e-300 ? Output_Layer_new[i] : 1e-300;
            }

            for (int i = 0; i < 10; i++ ) {
                //std::cout << Output_Layer_new[i] << std::endl;
            }

            
            //Output Vector
            for (int i = 0; i < 10; i++) {
                if (i == int(label)) {
                    RealOutput[i] = 1;
                }
                else {
                    RealOutput[i] = 0;
                }
            }
            
            //BackPropagation 

            double Loss = 0;
            for (int i = 0; i < 10; i++) {
                if (RealOutput[i] != 0) {
                    Loss = Loss - RealOutput[i] * log10(Output_Layer_new[i]);
                }
            }
            std::cout <<"Loss = " << Loss << std::endl;
            
            // 
            

            

            for (int i = 0; i < 10; i++) {
                dlBYdonew[i] = -RealOutput[i] / (Output_Layer_new[i] );
                
            }
            
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 10; j++) {
                    if (i == j) {
                        donewBYdo[i][j] = Output_Layer_new[i] - Output_Layer_new[i] * Output_Layer_new[i];
                    }
                    else
                        donewBYdo[i][j] = -Output_Layer_new[i] * Output_Layer_new[j];
                    

                }
            }

            doBYdw2[0] = 1;
            for (int i = 1; i < 17; i++) {
                doBYdw2[i] = Hidden_Layer1new[i-1];
            }
            
            
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 10; j++) {
                    tempdlBYdw1[i] = tempdlBYdw1[i]+ dlBYdonew[j] * donewBYdo[j][i];
                  
                }
            }

            

            

            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 17; j++) {
                    dlBYdw2[i][j] = tempdlBYdw1[i] * doBYdw2[j];
                }
            }


            

            double* dmiBYdwj = new double[785];
            dmiBYdwj[0] = 1;
            for (int i = 1; i < 785; i++) {
                dmiBYdwj[i] = (static_cast<double>(static_cast<unsigned char>(pixels[i - 1]))) / 256.0f;
            }


            
            for (int i = 0; i < 16; i++) {
                double temp2 = 0;
                for (int j = 0; j < 10; j++) {
                    double temp1=0;
                    for (int k = 0; k < 10; k++) {
                        temp1 = temp1 + donewBYdo[j][k] * W2[k][i + 1];
                    }
                    temp2 = temp2 + dlBYdonew[j] * temp1;
                }
                tempdlBYdw2[i] = temp2 * DerivLeaky(Hidden_Layer1[i]);
            }

            
            

            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 785; j++) {
                    dlBYdw1[i][j] = tempdlBYdw2[i] * dmiBYdwj[j];
                }
            }





            double alpha = 0.1;
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 17; j++) {
                    //W2[i][j] =  ((W2[i][j] - alpha * dlBYdw2[i][j]));
                    //std::cout << W2[i][j] << " ";
                }
                //std::cout << std::endl;
            }

            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 785; j++) {
                    W1[i][j] =  ((W1[i][j] - alpha * dlBYdw1[i][j]));
                    //std::cout << W1[i][j] << " ";

                }
                //std::cout << std::endl;
            } 
            


            

        }
        /*
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 17; j++) {
                W2[i][j] = W2new[i][j];
            }
        }

        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 785; j++) {
                W1[i][j] = W1new[i][j];
            }
        }
        */

    }

    saveArrayToFile("W1.txt", W1, 16, 785);
    saveArrayToFile("W2.txt", W2, 10, 17);

    std::cout << "Done" << std::endl;
    


    delete[] pixels;
    delete[] Hidden_Layer1;
    delete[] Hidden_Layer1new;
    delete[] Output_Layer;
    delete[] Output_Layer_new;
    delete[] RealOutput;

    for (int i = 0; i < 16; i++) {
        delete[] W1[i];
    }
    delete[] W1;

    for (int i = 0; i < 10; i++) {
        delete[] W2[i];
    }
    delete[] W2;

    
    delete[] dlBYdonew;
    delete[] doBYdw2;
    

   
    for (int i = 0; i < 10; i++) {
        delete[] donewBYdo[i];
        delete[] dlBYdw2[i];
    }

    for (int i = 0; i < 16; i++) {
        delete[] dlBYdw1[i];
    }

    
    delete[] donewBYdo;
    delete[] dlBYdw2;
    delete[] dlBYdw1;
}



void test_mnist(const char* image_filename, const char* label_filename, const char* W1_filename, const char* W2_filename) {
    // Load weights from files
    double** W1_test = new double* [16];
    double** W2_test = new double* [10];

    for (int i = 0; i < 16; i++) {
        W1_test[i] = new double[785];
    }

    for (int i = 0; i < 10; i++) {
        W2_test[i] = new double[17];
    }

    std::ifstream W1_file(W1_filename);
    std::ifstream W2_file(W2_filename);

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 785; j++) {
            W1_file >> W1_test[i][j];
        }
    }

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 17; j++) {
            W2_file >> W2_test[i][j];
        }
    }

    W1_file.close();
    W2_file.close();

    // Load test dataset
    std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);

    // ... (rest of your existing code)

    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;


    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2051) {
        std::cout << "Incorrect image file magic: " << magic << std::endl;
        return;
    }

    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2049) {
        std::cout << "Incorrect image file magic: " << magic << std::endl;
        return;
    }

    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    if (num_items != num_labels) {
        std::cout << "image file nums should equal to label num" << std::endl;
        return;
    }

    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    //std::cout << "image and label num is: " << num_items << std::endl;
    //std::cout << "image rows: " << rows << ", cols: " << cols << std::endl;

    char label;
    char* pixels = new char[rows * cols]; 

    int totalCorrect = 0;

    for (int kk = 0; kk < num_items; kk++) {

        //std::cout << num_items << " " << num_labels << std::endl;

        image_file.read(pixels, rows * cols);
        label_file.read(&label, 1);

        double* tempdlBYdw1 = new double[10];
        double* tempdlBYdw2 = new double[16];
        std::fill(tempdlBYdw1, tempdlBYdw1 + 10, 0.0);
        std::fill(tempdlBYdw2, tempdlBYdw2 + 16, 0.0);

        // calculating Layer 1
        double* Hidden_Layer1 = new double[16];
        double* Hidden_Layer1new = new double[16];
        double* Output_Layer = new double[10];
        double* Output_Layer_new = new double[10];
        double* RealOutput = new double[10];


        for (int i = 0; i < 16; i++) {
            Hidden_Layer1[i] = W1_test[i][0];
            for (int j = 1; j < 785; j++) {
                int pixelValue = static_cast<int>(static_cast<unsigned char>(pixels[j - 1]));
                Hidden_Layer1[i] = Hidden_Layer1[i] + W1_test[i][j] * ((double)pixelValue / 256.0f);
            }
            Hidden_Layer1new[i] = LeakyRelu(Hidden_Layer1[i]);
        }

        // calculating Layer 2

        double SoftSum = 0;
        // calculating Output
        for (int i = 0; i < 10; i++) {
            Output_Layer[i] = W2_test[i][0];
            for (int j = 1; j < 17; j++) {
                Output_Layer[i] = Output_Layer[i] + W2_test[i][j] * Hidden_Layer1new[j - 1];
            }
            SoftSum = SoftSum + exp(Output_Layer[i]);
        }


        for (int i = 0; i < 10; i++) {
            Output_Layer_new[i] = exp(Output_Layer[i]) / SoftSum;
            //Output_Layer_new[i] = Output_Layer_new[i] >= 1e-300 ? Output_Layer_new[i] : 1e-300;
        }

        int maximum = 0;
        for (int i = 0; i < 10; i++) {

            if (Output_Layer_new[i] > Output_Layer_new[maximum])
                maximum = i;
        }
        std::cout << "Our Predicted" << maximum << std::endl;
        std::cout << "Real Output" << int(label) << std::endl;

        if (maximum == int(label)) {
            totalCorrect++;
        }



        delete[] Hidden_Layer1;
        delete[] Hidden_Layer1new;
        delete[] Output_Layer;
        delete[] Output_Layer_new;
        delete[] RealOutput;

    }
    std::cout << "Total Correct Predicted = " << totalCorrect << std::endl;
    std::cout << "Efficiency = " <<(float(totalCorrect)/float(num_items))*100 << std::endl;
    // Deallocate memory for W1
    


    
}

int main() {
    srand(time(0));
    
    read_mnist_cv("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

    test_mnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", "W1.txt", "W2.txt");



    return 0;
}
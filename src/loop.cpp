/*
let's add c++ module one at a time --- so let's add the for loop first

==========To verify correctness
compare dimo_manifold.txt contents of python and c++ code on same input

*/

#include <iostream>
#include <fstream>
//#include <chrono>
//using namespace std::chrono;
using namespace std;

int main(int argc, char* argv[])
{
    FILE *fid;
    long dipha_identifier, diagram_identifier, num_pairs, bverts;
    double everts, pers;
    ofstream file1;

    //for (int i = 0; i < argc; ++i)
        //cout << argv[i] << "\n";

    fid = fopen(argv[1], "rb");

    fread(&dipha_identifier, sizeof(long), 1, fid);
    fread(&diagram_identifier, sizeof(long), 1, fid);
    fread(&num_pairs, sizeof(long), 1, fid);

    //cout << num_pairs << endl ;

    //auto start = high_resolution_clock::now();
    file1.open(argv[2]);
    for (int idx = 0; idx < num_pairs; idx++)
    {
        fread(&bverts, sizeof(long), 1, fid);
        fread(&everts, sizeof(double), 1, fid);
        fread(&pers, sizeof(double), 1, fid);

        file1 << bverts << '\t' << everts << '\t' << pers << '\n';
    }
    file1.close();
    //auto stop = high_resolution_clock::now();
    //auto duration = duration_cast<seconds>(stop - start);
    //cout << "time taken: " << duration.count() << " seconds" << endl;
    return 0;
}
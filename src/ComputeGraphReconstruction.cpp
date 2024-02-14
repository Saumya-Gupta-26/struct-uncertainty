#include<cstdlib>
#include <ctime>
#include <map>
#include <vector>
#include <iostream>
#include <queue>
#include <fstream>

using namespace std;

vector<int> bfs(int vert, int &branch_min, vector<int> vert_arr[], bool visited[], const vector<int> neighborhoods[], int pre[]){
			clock_t d_start = clock();
    	
	queue<int> q;
    	vector<int> visited_list;
	visited_list.clear();
	visited_list.push_back(vert);
	pre[vert] = -1;
    	q.push(vert);

	//cout << "entering queue loop" << endl;
	//branch_min = vert;
	branch_min = vert;

	int calls = 0;

	while (!q.empty())
	{
		calls++;
		int current_vert = q.front();
		q.pop();
	
		bool is_visited = visited[current_vert];	
		if (!is_visited)
		{
			clock_t d_start = clock();
			visited[current_vert] = true;
			clock_t d_end = clock();
			//cout << "discovered: " << (d_end - d_start) / float(CLOCKS_PER_SEC) << endl;
		}

		int current_x = vert_arr[current_vert][0];
		int current_y = vert_arr[current_vert][1];
		int current_z = vert_arr[current_vert][2];
		int current_val = vert_arr[current_vert][3];
		if (current_val < vert_arr[branch_min][3])
		{
			clock_t m_start = clock();
			branch_min = current_vert;
			clock_t m_end = clock();
			//cout << "min: " << (m_end - m_start) / float(CLOCKS_PER_SEC) << endl;
		}
		else if (current_val == vert_arr[branch_min][3] && current_z < vert_arr[branch_min][2])
		{
			branch_min = current_vert;
		}
		else if (current_val == vert_arr[branch_min][3] && current_z == vert_arr[branch_min][2] && current_y < vert_arr[branch_min][1])
		{
			branch_min = current_vert;
		}
		else if (current_val == vert_arr[branch_min][3] && current_z == vert_arr[branch_min][2] && current_y == vert_arr[branch_min][1] && current_x < vert_arr[branch_min][0])
		{
			branch_min = current_vert;
		}

		vector<int> neighborhood = neighborhoods[current_vert];

		for (int i = 0; i < neighborhood.size(); i++)
		{
			//cout << "working on neighbor " << i << endl;
			int neighbor = neighborhood[i];
			if (!visited[neighbor])
			{
			clock_t m_start = clock();
				visited[neighbor] = true;
				visited_list.push_back(neighbor);
				pre[neighbor] = current_vert;
				q.push(neighbor);
			clock_t m_end = clock();
			//cout << "min: " << (m_end - m_start) / float(CLOCKS_PER_SEC) << endl;
			}
		}
	}
	for (int i = 0; i < visited_list.size(); i++)
	{
		visited[visited_list[i]] = false;
	}
	//cout << "number of calls: " << calls << endl;
			clock_t d_end = clock();
			//cout << "whole call: " << (d_end - d_start) / float(CLOCKS_PER_SEC) << endl;
	return visited_list;
}

void retrieve_path(int vert, vector<int> &vPath, int pre[]){
    vPath.clear();
    //cout << "starting at vert " << vert << endl;
    while(pre[vert] != -1){
        vPath.push_back(vert);
        int vpre = pre[vert];
	//cout << vert << " " << vpre << endl;
        vert = vpre;
    }
    vPath.push_back(vert);
}

int main(int argc, char* argv[])
{
	string input_vert_filename = argv[1];
	string input_edge_filename = argv[2];
	int persistence_threshold = atoi(argv[3]);
	string output_dir = argv[4];

	vector<vector<int> > verts;
	verts.clear();

	ifstream fin;
        int x, y, z;
	double f;
        fin.open(input_vert_filename.c_str());
        // x y density
        // cout << "reading in verts from: " << input_vert_filename << endl;
        while (fin >> x >> y >> z >> f) {
		//cout << "vert: " << x << " " << y << " " << z << " " << f << endl;
                vector<int> vert;
		vert.clear();
		vert.push_back(x);
		vert.push_back(y);
		vert.push_back(z);
		vert.push_back(f);
		verts.push_back(vert);
        }
        fin.close();
	// std::cout << "read in " << verts.size() << " verts." << std::endl;

	vector<int>* vert_arr = new vector<int>[verts.size()];
	// std::cout << "Declared" << std::endl;
	for (int i = 0; i < verts.size(); i++)
	{
		vert_arr[i] = verts[i];
	}

	vector<vector<int> > edges;
	edges.clear();
	int u, v, i;
        fin.open(input_edge_filename.c_str());
        // u v i (u, v) starting from 0.
        // cout << "reading in edges from: " << input_edge_filename << endl;
	int cnt = 0;
        while (fin >> u >> v >> i) {
		/*
                if (i != 0 && i != -1)
		{
			cout << "edge " << cnt << ": " << u << " " << v << " " << i << endl;
		}
		*/
		cnt++;
        	vector<int> edge;
		edge.clear();
		edge.push_back(u);
		edge.push_back(v);
		edge.push_back(i);
		edges.push_back(edge);
	}
        fin.close();
	// std::cout << "read in " << edges.size() << " edges." << std::endl;

        // cout << "Computing vector field" << endl;
        vector<int>* neighborhoods = new vector<int>[verts.size()];
        // neighborhoods.clear();
        for (int i = 0; i < verts.size(); i++)
        {
                vector<int> neighborhood;
                neighborhood.clear();
                neighborhoods[i] = neighborhood;

        }

	// std::cout << "initialized neighbors" << std::endl;

	vector<vector<int> > vf_edges;
        vf_edges.clear();
        int ve_in_vf = 0;
        for (int i = 0; i < edges.size(); i++)
        {
		vector<int> edge = edges[i];
                int persistence = edge[2];
		if (persistence < 0)
		{
			persistence = -(persistence + 1);
		}
		else
		{
			continue;
		}
		if (persistence > persistence_threshold) //Xiaoling: Why?
		{
			continue;
		}
		//cout << "Edge " << i << " is in vector field" << endl;	
		/*
		int type = edge[3];
                if (persistence > persistence_threshold or type == 1)
                {
                        continue;
                }
		*/

                ve_in_vf++;
                //cout << i << endl;

                int v0 = edge[0];
                int v1 = edge[1];
                vector<int> field_edge;
                field_edge.clear();
                field_edge.push_back(v0);
                field_edge.push_back(v1);
                vf_edges.push_back(field_edge);
                neighborhoods[v0].push_back(v1);
                neighborhoods[v1].push_back(v0);
        }

	/*
	for (int i = 0; i < verts.size(); i++)
	{
		cout << i << "th neighborhood: ";
		for (int j = 0; j < neighborhoods[i].size(); j++)
		{
			cout << neighborhoods[i][j] << " ";
		}
		cout << endl;
	}
	*/

	// cout << "edges in vector field: " << ve_in_vf << endl;

	// cout << "Computing manifold" << endl;
        vector<int> min_computed;
        min_computed.clear();

        bool* visit = new bool[verts.size()];

        int* next_in_path = new int[verts.size()];

        vector<vector<int> > manifold;
        manifold.clear();

        for (int i = 0; i < verts.size(); i++)
        {
                min_computed.push_back(-1);
                visit[i] = false;
                next_in_path[i] = -1;
        }


	int know_min = 0;
        int not_know_min = 0;
        int critical_count = 0;
        for (int i = 0; i < edges.size(); i++)
        {
		// cout << "working on edge " << i << " out of " << edges.size() << endl;
		//clock_t d_start = clock();

		vector<int> edge = edges[i];
                int persistence = edge[2];
		if (persistence < 0)
                {
                        persistence = -(persistence + 1);
                }

                //cout << "persistence: " << ve_persistence[i] << ' ' << et_persistence[i] << endl;
                //int persistence = ve_per;
                if (persistence <= persistence_threshold)
                {
			/*
			clock_t d_end = clock();
			cout << "below thresh- finished immediately at  " << (d_end - d_start) / float(CLOCKS_PER_SEC) << endl;
                        */
			continue;
                }

                critical_count++;

		//clock_t start = clock();
                vector<int> critical_edge;
                critical_edge.clear();
                critical_edge.push_back(edge[0]);
                critical_edge.push_back(edge[1]);
                manifold.push_back(critical_edge);

		for (int j = 0; j < 2; j++)
                {
                        //cout << "working on " << j << "th vert" << endl;
                        int v = edge[j];
                        vector<int> vPath;
                        vPath.clear();
                        if (min_computed[v] == -1)
                        {
				//cout << "min not known" << endl;
                                not_know_min++;
                                //cout << "have not computed yet" << endl;
                                int branch_min;
                                vector<int> component = bfs(v, branch_min, vert_arr, visit, neighborhoods, next_in_path);
				for (int k = 0; k < component.size(); k++)
                                {
                                        min_computed[component[k]] = branch_min;
                                }

                                //cout << "component size: " << component.size() << endl;
                                //cout << "minimum: " << branch_min << endl;
                                bfs(branch_min, branch_min, vert_arr, visit, neighborhoods, next_in_path);
                                retrieve_path(v, vPath, next_in_path);
                        }
                        else
                        {
				//cout << "min known" << endl;
                                know_min++;
                                retrieve_path(v, vPath, next_in_path);
                        }
                        manifold.push_back(vPath);
			/*
			cout << "VPATH: ";
			for (int k = 0; k < vPath.size(); k++)
			{
				cout << vPath[k] << " ";
			}
			cout << endl;
			*/
                }
                /*
		clock_t d_end = clock();
		cout << "finished at  " << (d_end - d_start) / float(CLOCKS_PER_SEC) << endl;
		*/
	}

	delete[] neighborhoods;
	delete[] vert_arr;

	// cout << "outputting..." << endl;

        vector<int> output_indices;
        output_indices.clear();
        for (int i = 0; i < verts.size(); i++)
        {
                output_indices.push_back(-1);
        }

        int output_index = 0;
        vector<int> output_verts;
        output_verts.clear();
        vector<vector<int> > output_edges;
        output_edges.clear();

	for (int i = 0; i < manifold.size(); i++)
        {
                vector<int> component = manifold[i];
                for (int j = 0; j < component.size() - 1; j++)
                {
                        //cout << "beginning of i loop: " << output_index << endl;
                        int v0 = component[j];
                        int ov0;
                        if (output_indices[v0] != -1)
                        {
                                ov0 = output_indices[v0];
                        }
                        else
                        {
                                ov0 = output_index;
                                output_indices[v0] = output_index;
                                //cout << output_index << ": " << v0 << endl;
                                output_verts.push_back(v0);
                                output_index = output_index + 1;
                        }
                        //cout << "after v0: " << output_index << endl;

                        int v1 = component[j + 1];
                        int ov1;
			if (output_indices[v1] != -1)
                        {
                                //cout <<"t1" << endl;
                                ov1 = output_indices[v1];
                        }
                        else
                        {
                                //cout << ov1 << " ";
                                ov1 = output_index;
                                //cout << ov1 << " ";
                                output_indices[v1] = output_index;
                                //cout << output_indices[v1] << endl;
                                //cout << output_index << ": " << v1 << endl;
                                output_verts.push_back(v1);
                                output_index = output_index + 1;
                        }

                        vector<int> edge;
                        edge.clear();
                        edge.push_back(ov0);
                        edge.push_back(ov1);
                        output_edges.push_back(edge);
                }
        }

        // cout << "writing files" << endl;

	string vertex_filename = output_dir + "dimo_vert.txt";
        ofstream vFile(vertex_filename.c_str());
        for (int i = 0; i < output_verts.size(); i++)
        {
                vector<int> vert = verts[output_verts[i]];
                vFile << vert[0] << " " << vert[1] << " " << vert[2] << " " << vert[3] << endl;
        }

        string edge_filename = output_dir + "dimo_edge.txt";
        ofstream eFile(edge_filename.c_str());
        for (int i = 0; i < output_edges.size(); i++)
        {
                vector<int> edge = output_edges[i];
                //cout << edge[0] << " " << edge[1] << endl;
                eFile << edge[0] << " " << edge[1] << endl;
        }

        return 0;

}


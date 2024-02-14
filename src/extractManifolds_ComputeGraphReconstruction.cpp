#include<cstdlib>
#include <ctime>
#include <map>
#include <vector>
#include <iostream>
#include <queue>
#include <fstream>

using namespace std;

/*
Compilation command:
g++ -o manifold.out extractManifolds_ComputeGraphReconstruction.cpp

Saving the output file with a new line after every manifold

Cubical complex: cubical complex (3D grid), where each vertex corresponds to a voxel in the input image and has a density (intensity) value. In 2D, we have vertices, edges, sqaures. Edge is between 2 vertices. They use lower-star filtration, that is, assign to each simplex the value of its maximum vertex. 
*/

/*
Reference papers:
[1] Graph reconstruction via discrete Morse theory
[2] Detection and skeletonization of single neuron using topo methods
*/


/*
what does this bfs function do? 
Inputs-
- vert: vertex from which bfs starts.
- branch_min: It is sent by reference and its value is set in this function. It contains vertex-ID of the vertex with minimum intensity value (lexicographically) that was visited in this function call.
- vert_arr: info about all vertices (x,y,z,intensity).
- visited: visited array used in BFS.
- neighborhoods: neighborhood of each vertex; if vertex vi has a neighbor vn, then it means the edge vi-vn is a critical negative edge which passed the persistence-threshold check. In a way, the neighborhood gives guidance on the flow of growing a path. We don't blindly apply BFS on all 4-connectivity (2D case) neighbors, instead, we only push to BFS queue q these specific neighbors.
- pre: keeps track of parent vertex p of each vertex v (parent p is the one who pushed vertex v into the BFS queue q).

Returns-
- branch_min: Value updated in this function
- visited_list: Contains all the vertices that were visited in this function call. The BFS terminates if all vertices reachable (from vert via neighbors) have been visited. So we kinda made a "branch"/"component"/"tree"/"structure", that's why branch_min is named that way.
*/
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
		int current_val = vert_arr[current_vert][3]; //saum: intensity value of vertex/voxel

                //saum: branch_min has lowest intensity value, and in case of tie, it is lexicographically lower 
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
				pre[neighbor] = current_vert; //saum: pre keeps track of parent
				q.push(neighbor);
			clock_t m_end = clock();
			//cout << "min: " << (m_end - m_start) / float(CLOCKS_PER_SEC) << endl;
			}
		}
	}

        //saum: whatever was visited (aka visited[]=true) in this function is reset back to false (taking help of visited_list to keep track of what was set to true)
	for (int i = 0; i < visited_list.size(); i++)
	{
		visited[visited_list[i]] = false;
	}
	//cout << "number of calls: " << calls << endl;
			clock_t d_end = clock();
			//cout << "whole call: " << (d_end - d_start) / float(CLOCKS_PER_SEC) << endl;
	return visited_list; //all the vertices that were visited in this function are returned.
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
        //cout << "reading in verts from: " << input_vert_filename << endl;
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
	//std::cout << "read in " << verts.size() << " verts." << std::endl;

	vector<int>* vert_arr = new vector<int>[verts.size()];
	//std::cout << "Declared" << std::endl;
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
	//std::cout << "read in " << edges.size() << " edges." << std::endl;

        //cout << "Computing vector field" << endl;
        //saum: each vertex is associated with a cubic neighborhood; initially, the neighborhood is empty (neighborhood.clear();)
        vector<int>* neighborhoods = new vector<int>[verts.size()];
        // neighborhoods.clear();
        for (int i = 0; i < verts.size(); i++)
        {
                vector<int> neighborhood;
                neighborhood.clear();
                neighborhoods[i] = neighborhood;

        }

	//std::cout << "initialized neighbors" << std::endl;

	vector<vector<int> > vf_edges; 
        /*
        saum: vf_edges : edges in the vector field (i think it's the DGVF); It is computed but not directly used anywhere, because neighborhoods contains similar info, and neighborhoods is used later on.
        Basically, this collects all the negative edges which FAIL the persistence-threshold test/check. The edges which fail the test will be part of the unstable manifold created by other edges, that is, the edges which fail the test will not be the originators of the unstable manifold. Since the unstable manifold is a vertex-edge path, we need to define the neighborhood of each vertex --- the neighborhood tells which edge can be used to hop to the next vertex. Thus the BFS later on is not euclidean shortest distance using 4-connectivity, rather, the BFS will traverse only these edges which failed the test.
        */
        vf_edges.clear();
        int ve_in_vf = 0; //saum: ve_in_vf counts the number of edges in the vector field
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
		if (persistence > persistence_threshold)
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

                //saum: an edge is in the DGVF if it passes the persistence-threshold check (see above few lines); Thus, an edge passing the persistence-threshold is added to vf_edges vector. Furthermore, the endpoints of this edge --- v0 and v1 --- they both are added into each others neighborhoods. In cubical complex, edges are between vertices which differ by 1 in exactly one direction (aka no diagonal edges, aka 4-connectivity). Thus, max size of a neighborhood of a vertex is 4.
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

	//cout << "edges in vector field: " << ve_in_vf << endl;

	//cout << "Computing manifold" << endl;
        vector<int> min_computed;
        min_computed.clear();

        bool* visit = new bool[verts.size()];

        int* next_in_path = new int[verts.size()];

        vector<vector<int> > manifold; //saum: one manifold = collection of edges (one edge is collection of vertices and so we have vector<vector>)
        vector<int> pers_manifold; //saum: this variable added by me; stores the persistence value of each corresponding edge in the manifold
        pers_manifold.clear();
        manifold.clear();

        for (int i = 0; i < verts.size(); i++)
        {
                min_computed.push_back(-1);
                visit[i] = false;
                next_in_path[i] = -1;
        }


	int know_min = 0; //saum: count computed but not used
        int not_know_min = 0; //saum: count computed but not used
        int critical_count = 0;
        /* 
        saum: Going through all edges again and using the persistence-check. This time, we conside the edges which PASS the persistence-threshold check. These edges will be the creators of the unstable manifold.
        
        Now, the unstable manifold is the path from these edges to critical vertices. Consider one of these edges 'e'. The critical vertex is the vertex having lowest intensity value (branch_min) in the tree that starts from one endpoint of 'e'. Since 'e' has 2 endpoints, we have 2 branch_min and thus we have two VPaths --- from each endpoint to their corresponding branch_min/sink. Union of these two VPaths with 'e' makes up the manifold. 

        Notice that the VPaths always go through the vf_edges defined earlier. This is achieved by using the neighborhoods variable in the BFS code. See Section 3 in paper [1].
        */
        for (int i = 0; i < edges.size(); i++)  
        {
		//cout << "working on edge " << i << " out of " << edges.size() << endl;
		//clock_t d_start = clock();

		vector<int> edge = edges[i];
                int persistence = edge[2];
		if (persistence < 0)
                {
                        persistence = -(persistence + 1);
                }

                //cout << "persistence: " << ve_persistence[i] << ' ' << et_persistence[i] << endl;
                //int persistence = ve_per;
                if (persistence <= persistence_threshold) //saum: ignoring edges which FAIL the persistence-threshold check
                {
			/*
			clock_t d_end = clock();
			cout << "below thresh- finished immediately at  " << (d_end - d_start) / float(CLOCKS_PER_SEC) << endl;
                        */
			continue;
                }

                //saum: At this point, the edge has passed persistence-threshold check.
                critical_count++;

		//clock_t start = clock();
                vector<int> critical_edge;
                critical_edge.clear();
                critical_edge.push_back(edge[0]); //saum: v0
                critical_edge.push_back(edge[1]); //saum: v1
                manifold.push_back(critical_edge); 
                pers_manifold.push_back(persistence); //saum: Manifold contains the critical edge; Below we will also include the VPath generated from this critical edge into the Manifold. 

		for (int j = 0; j < 2; j++) //saum: v0 and v1 of this edge will be processed
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
                                int branch_min; //saum: sent by reference to bfs and so will be populated there
                                vector<int> component = bfs(v, branch_min, vert_arr, visit, neighborhoods, next_in_path);
				for (int k = 0; k < component.size(); k++)
                                {
                                        min_computed[component[k]] = branch_min;
                                        //saum: BFS is done once and we get component. If we do BFS starting from any vertex in this component, we will always get the same component and branch_min. Thus we update branch_min for all vertices in the component. And later, we do not have to redo BFS for this tree (in a forest; all disjoint paths in BFS are trees making up a forest), rather, we can just directly use branch_min.
                                }

                                //cout << "component size: " << component.size() << endl;
                                //cout << "minimum: " << branch_min << endl;

                                /* 
                                saum: calling bfs again but this time bfs starts from branch_min, and NOT from the critical_edge endpoint v.
                                We know v is reachable from branch_min because branch_min was found when BFS started from v.
                                This branch_min is the corresponding "sink" of this critical_edge. From Section 3 in paper [1], they say this sink is the actual critical vertex. The paths from the endpoints of the critical_edge to the corresponding sink (critical vertex) is the unstable manifold. And notice, the path goes through the edges that failed the persistence-threshold check.
                                
                                Thus we see in retrieve_path, we recover the path from each endpoint separately. The path is recovered with the help of next_in_path which was populated with the parent vertex in the BFS code.
                                Since BFS starts from branch_min this time, pre[branch_min] = next_in_path[branch_min] = -1

                                In retrieve_path, we terminate the code when we reach pre[] = -1 ; Thus, we get the path (aka vPath) from v to branch_min
                                */
                                bfs(branch_min, branch_min, vert_arr, visit, neighborhoods, next_in_path);
                                retrieve_path(v, vPath, next_in_path);
                        }
                        else
                        {
                                //saum: this vertex is already a part of a BFS component/branch/tree/structure, and so we directly call retrieve_path
				//cout << "min known" << endl;
                                know_min++;
                                retrieve_path(v, vPath, next_in_path);
                        }
                        manifold.push_back(vPath); //saum: vPath contains multiple vertices (it's the branch/component/tree/structure); This vPath is the pi(u) in Algorithm 2 (Line 3 and 4) in paper [1].
                        pers_manifold.push_back(persistence);
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

	//cout << "outputting..." << endl;

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

        //saum : check that manifold and pers_manifold have same number of elements!
        if (manifold.size() != pers_manifold.size())
        {
                //cout << "manifold " << manifold.size() << " and pers_manifold " << pers_manifold.size() << " have different lengths!!";
                return 1; //error
        }

        /* 
        saum : I am writing manifold edges to another output file called dimo_manifold.txt. Check if it's contents match dimo_edge.txt 

        From paper  : For each edge, the 1-unstable manifold is equivalent to the union of the edge with the paths from both vertices to the sink of their corresponding tree.
        The variable manifold in this code separately contains all the components pi(u),pi(v),{e} (From paper [1] Algorithm 2 Line 4). TODO: I need to merge these 3 into one when writing to dimo_manifold.txt. You'll notice that the number of manifolds (first line in dimo_manifold.txt) is always a multiple of 3. This is because manifold.push_back is called thrice in each run of the for-loop above.
        */
        string manifold_filename = output_dir + "dimo_manifold.txt";
        ofstream mFile(manifold_filename.c_str());
        mFile << manifold.size()/3 << endl; //saum: first line of file contains total number of manifolds; dividing by 3 because taking union of edge & vpaths from both its endpoints 
	for (int i = 0; i < manifold.size(); i++)
        {
                if (i%3 == 0)
                        mFile << endl; //saum : adding new line to separate manifolds

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
                        edge.push_back(pers_manifold[i]); //saum
                        output_edges.push_back(edge);
                        mFile << ov0 << " " << ov1 << " " << pers_manifold[i] << endl; //saum : writing manifold edge with persistence
                }
        }

        mFile << endl << "EOF" << endl; //saum : writing EOF for easier computation by other codes
        //cout << "writing files" << endl;

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
                eFile << edge[0] << " " << edge[1] << " " << edge[2] << endl; //saum : writing persistence
        }

        return 0;

}


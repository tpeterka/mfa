//--------------------------------------------------------------
// Extension of an MFA-DIY block to read the NASA 3d 
// retropropulsion data set with a swap-reduce
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------
#ifndef _MFA_NASA_BLOCK
#define _MFA_NASA_BLOCK

#include <diy/point.hpp>

#include "block.hpp"

using namespace std;

// NASA block
template <typename T>
struct NASABlock : public BlockBase<T>
{
    using Base = BlockBase<T>;
    using Base::dom_dim;
    using Base::pt_dim;
    using Base::core_mins;
    using Base::core_maxs;
    using Base::bounds_mins;
    using Base::bounds_maxs;
    using Base::input;
    using Base::approx;
    using Base::errs;
    using Base::mfa;
    // using Base::ndom_outpts;
    // using Base::blend;

    using Point = diy::Point<T, 4>;

    vector<int64_t> v_gids;
    vector<Point> points;
    VectorX<T> box_mins;
    VectorX<T> box_maxs;

    static
        void* create()              { return mfa::create<NASABlock>(); }

    static
        void destroy(void* b)       { mfa::destroy<NASABlock>(b); }
    static
        void add(                                   // add the block to the decomposition
            int                 gid,                // block global id
            const Bounds<T>&    core,               // block bounds without any ghost added
            const Bounds<T>&    bounds,             // block bounds including any ghost region added
            const Bounds<T>&    domain,             // global data bounds
            const RCLink<T>&    link,               // neighborhood
            diy::Master&        master,             // diy master
            int                 dom_dim,            // domain dimensionality
            int                 pt_dim,             // point dimensionality
            T                   ghost_factor = 0.0) // amount of ghost zone overlap as a factor of block size (0.0 - 1.0)
    {
        mfa::add<NASABlock, T>(gid, core, bounds, domain, link, master, dom_dim, pt_dim, ghost_factor);

        // Immediately update the new block (which has the given gid)
        // Copies domain into box_mins/maxs, which will be used for swap-reduce
        int lid = master.lid(gid);
        NASABlock<T>* b = master.get<NASABlock<T>>(lid);
        b->box_mins.resize(dom_dim);
        b->box_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            b->box_mins[i] = domain.min[i];
            b->box_maxs[i] = domain.max[i];
        }

        cerr << "core: " << b->core_mins.transpose() << "; " << b->core_maxs.transpose() << endl;
        cerr << "bounds: " << b->bounds_mins.transpose() << "; " << b->bounds_maxs.transpose() << endl;
    }
    static
        void save(const void* b_, diy::BinaryBuffer& bb)    { mfa::save<NASABlock, T>(b_, bb); }
    static
        void load(void* b_, diy::BinaryBuffer& bb)          { mfa::load<NASABlock, T>(b_, bb); }



    // Provide a container for native volume header data (for a single subdomain).
    struct NASAHeader {
        NASAHeader() : n_nodes_0(0), n_nodes(0), size(0) {}

        /// File format version.
        string version;
        /// FUN3D internal: number of grid points local to the partition.
        size_t n_nodes_0;
        /// Number of nodes (also known as vertices or grid points).
        size_t n_nodes;
        /// Names of variables, in the order they occur in the data.
        vector<string> variables;
        /// Map from local to global vertex numbering, 1-based.
        vector<int64_t> local_to_global;
        /// Header size, in bytes.
        size_t size;
    };

    string read_nasa_string(istream& is)
    {
        int32_t length;
        is.read((char*) &length, sizeof(int32_t));
        vector<char> chars(length);
        is.read(chars.data(), chars.size());
        return string(chars.data(), chars.size());
    }

    void read_nasa_volume_header(string data_filename, NASAHeader& h)
    {
        ifstream is(data_filename);
        if (!is)
        {
            cerr << "Failed to open data file" << endl;
            exit(1);
        }

        int32_t magic;
        is.read((char*) &magic, sizeof(int32_t));
        if (magic != 305419896)
        {
            cerr << "bad magic" << endl;
            exit(1);
        }

        int32_t i32;    

        h.version = read_nasa_string(is);

        is.read((char*) &i32, sizeof(int32_t));
        h.n_nodes_0 = (size_t) i32;

        is.read((char*) &i32, sizeof(int32_t));
        h.n_nodes = (size_t) i32;

        is.read((char*) &i32, sizeof(int32_t));
        size_t n_variables = (size_t) i32;

        h.variables.resize(n_variables);
        for (size_t i = 0; i < n_variables; ++i)
            h.variables[i] = read_nasa_string(is);

        h.local_to_global.resize(h.n_nodes);
        is.read((char*) h.local_to_global.data(), h.local_to_global.size() * sizeof(int64_t));

        h.size = is.tellg();

        cout << "Header info: " << '\n'
             << "|   Version: " << h.version << "\n"
             << "|   File Size (bytes): " << h.size << "\n"
             << "|   Nodes: " << h.n_nodes << "\n"
             << "|   Variable Names: ";
        for (int i = 0; i < h.variables.size(); i++)
        {
            cout << h.variables[i] << " ";
        }
        cout << "-----------------------" << endl;
    }

    size_t seek_time_step(istream& is, const NASAHeader& h, int ts)
    {
        int ts0 = -1;
        size_t size1 = sizeof(int32_t) + h.n_nodes * h.variables.size() * sizeof(float);
        size_t i = 0;

        // On first iteration, we check what the delta-T between the first and second
        // time steps is. Then, we use that delta-T to seek ahead to the appropriate
        // position. Note: The time steps are not sequential, so it could be that
        // ts0 = 1000, ts1 = 1200; then delta-T = 200.
        for ( ; ; i++) {
            int32_t tsi;
            is.seekg(h.size + i * size1);
            is.read((char*) &tsi, sizeof(int32_t));

            if (is.eof())
            {
                cerr << "Time step " << ts << " not found.\nExiting" << endl;
                exit(1);
            }

            if (tsi == ts)
                break;

            if (i == 0) 
            {
                ts0 = tsi;
            }
            else if (i == 1) 
            {
                //
                // Assuming snapshots saved at a fixed stride, calculate where
                // we would expect to find time step ts, based on ts0 and ts1,
                // and see if that is correct.  Seeking to and reading 4-byte
                // tsi values one at a time can be slow going if there are
                // many time steps saved to a file.  Use this heuristic to try
                // to find the matching time step faster.
                //
                int ts1 = tsi;
                int dt = ts1 - ts0;
                if (dt > 0 && (ts - ts0) > 0 && (ts - ts0) % dt == 0) 
                {
                    size_t ii = (size_t) ((ts - ts0) / dt);
                    is.seekg(h.size + ii * size1);
                    is.read((char*) &tsi, sizeof(int32_t));
                    if (!is.eof() && tsi == ts) {
                        i = ii;
                        break;
                    }
                    else if (is.eof())
                    {
                        cerr << "Reached end of file while looking for timestep " << ts << ".\nExiting." << endl;
                        exit(1);
                    }
                    else
                    {
                        cerr << "Timestep " << ts << " not found." << endl;
                        cerr << "  Closest timestep found: " << tsi << endl;
                        cerr << "Exiting." << endl;
                        exit(1);
                    }
                }
            }
        }
        return i;
    }

    void read_nasa3d_retro_mesh(string mesh_filename)
    {
        ifstream is(mesh_filename);
        if (!is)
        {
            cerr << "Failed to open mesh file.\nExiting." << endl;
            exit(1);
        }

        size_t n_nodes = -1;
        size_t n_surf_tris = -1;
        size_t n_surf_quads = -1;
        size_t n_tets = -1;
        size_t n_pyramids = -1;
        size_t n_prisms = -1;
        size_t n_hexs = -1;

        int32_t i32;    
        is.read((char*) &i32, sizeof(int32_t));
        n_nodes = (size_t) i32;

        is.read((char*) &i32, sizeof(int32_t));
        n_surf_tris = (size_t) i32;

        is.read((char*) &i32, sizeof(int32_t));
        n_surf_quads = (size_t) i32;

        is.read((char*) &i32, sizeof(int32_t));
        n_tets = (size_t) i32;

        is.read((char*) &i32, sizeof(int32_t));
        n_pyramids = (size_t) i32;

        is.read((char*) &i32, sizeof(int32_t));
        n_prisms = (size_t) i32;

        is.read((char*) &i32, sizeof(int32_t));
        n_hexs = (size_t) i32;

        cout << "Reading mesh file" << "\n"
             << "  Nodes:             " << n_nodes << "\n"
             << "  Surface Triangles: " << n_surf_tris << "\n"
             << "  Surface Quads:     " << n_surf_quads << "\n"
             << "  Tetrahedra:        " << n_tets << "\n"
             << "  Pyramids:          " << n_pyramids << "\n"
             << "  Prisms:            " << n_prisms << "\n"
             << "  Hexahedra:         " << n_hexs << endl;

        // read x,y,z coordinates of each node
        vector<float> xyz_coords(n_nodes * 3);
        is.read((char*) xyz_coords.data(), n_nodes * 3 * sizeof(float));
        vector<float> maxs = {xyz_coords[0], xyz_coords[1], xyz_coords[3]};
        vector<float> mins = {xyz_coords[0], xyz_coords[1], xyz_coords[3]};
        for (size_t i = 0; i < n_nodes; i++)
        {
            points[i][0] = xyz_coords[i * 3];
            points[i][1] = xyz_coords[i * 3 + 1];
            points[i][2] = xyz_coords[i * 3 + 2];

            for (int l = 0; l < 3; l++)
            {
                if (points[i][l] > maxs[l]) maxs[l] = points[i][l];
                if (points[i][l] < mins[l]) mins[l] = points[i][l];
            }
        }
        cout << ">>" << mesh_filename << " mins: " << mins[0] << " " << mins[1] << " " << mins[2] << "<<" << endl;
        cout << ">>" << mesh_filename << " maxs: " << maxs[0] << " " << maxs[1] << " " << maxs[2] << "<<" << endl;
    }

    void read_nasa3d_retro_data(string data_filename, int time_step, string var_name, NASAHeader& header)
    {
        ifstream is(data_filename);
        if (!is)
        {
            cerr << "Failed to open data file" << endl;
            exit(1);
        }

        // get variable index from name
        auto it = std::find(header.variables.begin(), header.variables.end(), var_name);
        if (it == header.variables.end())
        {
            cerr << "Variable \'" << var_name << "\' not in variable name list" << endl;
            exit(1);
        }
        int var_id = std::distance(header.variables.begin(), it);

        // Find start of requested time step in file
        seek_time_step(is, header, time_step);

        // Read variable data into points vector
        vector<float> tmp(header.n_nodes * header.variables.size());
        is.read((char*) tmp.data(), tmp.size() * sizeof(float));

        for (int i = 0; i < header.n_nodes; i++)
        {
            points[i][3] = tmp[i * header.variables.size() + var_id]; 
        }
    }

    void read_nasa3d_retro(const       diy::Master::ProxyWithLink& cp,
                        MFAInfo&    mfa_info,
                        DomainArgs& args,
                        int subdomain_id, int time_step, string var_name)
    {
        assert(mfa_info.nvars() == 1);
        assert(mfa_info.geom_dim() == 3);
        assert(mfa_info.model_dims()(1) = 1);

        const int nvars         = mfa_info.nvars();
        const int gdim          = mfa_info.geom_dim();
        const VectorXi mdims    = mfa_info.model_dims();

        cout << "Reading NASA 3d Retropopulsion Dataset" << endl;

        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        bounds_mins.resize(pt_dim);
        bounds_maxs.resize(pt_dim);

        subdomain_id = cp.gid() + 1;

        // Here we expect args.infile to be the path PREFIX to the root of the NASA data folder
        string mesh_filename = args.infile + "dAgpu0145_Fa_mesh.lb4." + to_string(subdomain_id);
        string data_filename = args.infile + "2000unsteadyiters/dAgpu0145_Fa_volume_data." + to_string(subdomain_id);

        // Read header and print some information
        NASAHeader header;
        read_nasa_volume_header(data_filename, header);
        points.resize(header.n_nodes);

        // Move global vertex ids into block (is this info used?)s
        v_gids = std::move(header.local_to_global);

        // args.tot_ndom_pts = header.n_nodes;
        // input = new mfa::PointSet<T>(dom_dim, mdims, args.tot_ndom_pts);

        read_nasa3d_retro_data(data_filename, time_step, var_name, header);

        // Read x,y,z coordinates
        read_nasa3d_retro_mesh(mesh_filename);
    }

    void set_input(const       diy::Master::ProxyWithLink& cp,
                        MFAInfo&    mfa_info,
                        DomainArgs& args)
    {
        input = new mfa::PointSet<T>(dom_dim, mfa_info.model_dims(), points.size());

        for (int j = 0; j < points.size(); j++)
        {
            for (int l = 0; l < pt_dim; l++)
            {
                input->domain(j, l) = points[j][l];
            }
        }
        cerr << "core extent:" << endl;
        cerr << "  mins: " << input->domain.leftCols(dom_dim).colwise().minCoeff() << endl;
        cerr << "  maxs: " << input->domain.leftCols(dom_dim).colwise().maxCoeff() << endl;
        input->set_domain_params(core_mins, core_maxs);



        // set bounds in each dimension
        // first dom_dim entries were set during block decomposition
        bounds_mins.tail(pt_dim-dom_dim) = input->domain.rightCols(pt_dim-dom_dim).colwise().minCoeff();
        bounds_maxs.tail(pt_dim-dom_dim) = input->domain.rightCols(pt_dim-dom_dim).colwise().maxCoeff();

        // initialize MFA models (geometry, vars, etc)
        this->setup_MFA(cp, mfa_info);

        // debug
        cerr << "data extent:\n";
        cerr << "  mins: " << mfa::print_vec(bounds_mins) << endl;
        cerr << "  maxs: " << mfa::print_vec(bounds_maxs) << endl;
        // cerr << "bounds extent:\n";
        // cerr << "  mins: "; mfa::print_vec(dom_mins);
        // cerr << "  maxs: "; mfa::print_vec(dom_maxs);
        cerr << endl;
    }


    //
    // callback function for redistribute operator, called in each round of the reduction
    //
    static
    void redistribute(NASABlock* b,                                 // local block
                    const diy::ReduceProxy& srp,              // communication proxy
                    const diy::RegularSwapPartners& partners) // partners of the current block
    {
        string logname = "redistribute.log." + to_string(srp.gid());
        ofstream log(logname);

        unsigned      round    = srp.round();                   // current round number
log << "round " << round << " gid " << srp.gid() << endl;
        // step 1: dequeue
        // dequeue all the incoming points and add them to this block's vector
        // could use srp.incoming() instead
        for (int i = 0; i < srp.in_link().size(); ++i)
        {
            int nbr_gid = srp.in_link().target(i).gid;
            if (nbr_gid == srp.gid())
                continue;

            std::vector<Point>    in_points;
            srp.dequeue(nbr_gid, in_points);
            log << "[" << srp.gid() << ":" << round << "] Received " << (int) in_points.size() << " points from [" << nbr_gid << "]" << endl;
            fmt::print(stderr, "[{}:{}] Received {} points from [{}]\n",
                    srp.gid(), round, (int) in_points.size(), nbr_gid);
            for (size_t j = 0; j < in_points.size(); ++j)
                b->points.push_back(in_points[j]);
        }
log << "  done dequeuing" << endl;
        // step 2: sort and enqueue
        if (srp.out_link().size() == 0)        // final round; nothing needs to be sent
            return;

        std::vector< std::vector<Point> > out_points(srp.out_link().size());
        int group_size = srp.out_link().size();  // number of outbound partners
        int cur_dim    = partners.dim(round);    // current dimension along which groups are formed
        // sort points into vectors corresponding to neighbor blocks
        for (size_t i = 0; i < b->points.size(); ++i) // for all points
        {
            auto loc = static_cast<size_t>(floor((b->points[i][cur_dim] - b->box_mins[cur_dim]) /
                                                (b->box_maxs[cur_dim] - b->box_mins[cur_dim]) * group_size));
            
            // If the point is exactly on the farthest boundary, the above formula can overshoot
            // and set loc==group_size. However, out_points has size=group_size, so this would cause an error
            // In this case we simply subtract 1 (thus sending this point to the closest partner)
            if (loc == group_size) loc--;

            // If, even after decrementing, we still have loc too big, then print an error
            if (loc > group_size-1)
            {
                cerr << "######" << loc << " " << group_size - 1 << endl;
                cerr << "##########" << b->points[i][0] << " " << b->points[i][1] << " " << b->points[i][2] << endl;
                cerr << "##########" << cur_dim << " " << b->box_mins[cur_dim] << " " << b->box_maxs[cur_dim] << endl;
            }
            out_points[loc].push_back(b->points[i]);
        }

log << "  done sorting" << endl;
        int pos = -1;
        // enqueue points to neighbor blocks
        for (int i = 0; i < group_size; ++i)     // for all neighbors
        {
            if (srp.out_link().target(i).gid == srp.gid())
            {
                b->points.swap(out_points[i]);
                pos = i;
            }
            else
            {
                srp.enqueue(srp.out_link().target(i), out_points[i]);
                log << "[" << srp.gid() << "] Sent " << (int) out_points[i].size() << " points to [" << srp.out_link().target(i).gid << "]" << endl;
                fmt::print(stderr, "[{}] Sent {} points to [{}]\n",
                        srp.gid(), (int) out_points[i].size(), srp.out_link().target(i).gid);
            }
        }
log << "  done enqueuing" << endl;


        // step 3: readjust box boundaries for next round
        float new_min = b->box_mins[cur_dim] + (b->box_maxs[cur_dim] -
                                            b->box_mins[cur_dim])/group_size*pos;
        float new_max = b->box_mins[cur_dim] + (b->box_maxs[cur_dim] -
                                            b->box_mins[cur_dim])/group_size*(pos + 1);

        // Need to handle floating point errors where 
        // box_mins + (box_maxs - box_mins) / group_size * group_size != box_maxs
        if (pos == group_size-1)
        {
            new_max = b->box_maxs[cur_dim];
        }
        b->box_mins[cur_dim] = new_min;
        b->box_maxs[cur_dim] = new_max;
    }
};

#endif
#ifndef PARTITION_HPP
#define PARTITION_HPP

// ---------------------------------------------------------------------------------------- //

struct Partition {
    int base;
    int local_grid_rows;
    int first_grid_row;
    int local_n;
    int global_offset;
    int up;
    int down;
    int n;
};

// ---------------------------------------------------------------------------------------- //

// fill partion struct 
Partition fill_part(int n, int rank, int size);

// ---------------------------------------------------------------------------------------- //

#endif
import time
from result import *
import numpy as np

INF = 1e12

def esdf_1d(grid):
    """Calculates the distance to the closest obstacle along the first axis

    Args:
        grid (float64): Holds the upper bound on the nearest obstacle distance along the axis
    """
    N = grid.shape[0]
    k = np.zeros(grid.shape[1:]).astype(np.int32) # Index of rightmost element in lower envelope
    v = np.ones(grid.shape)*INF # Locations of parabolas in lower envelope
    v[0,...] = 0
    z = np.zeros((N+1,) + grid.shape[1:]) # Location of boundaries between parabolas
    z[0,...] = -INF
    z[1,...] = INF
    
    # Note that the grid values (0 if obstacle, otherwise infty) if the function f in this case
    # The modifications made in this loop have also been made in the next loop
    for q in range(1, N):
        v_k = np.take_along_axis(v, k[None, :], axis=0).astype(np.int32) # Get value of v at corresponding k indices
        grid_v_k = np.take_along_axis(grid, v_k, axis=0) # Get the values of grid at the corresponding k values
        s = ((grid[q,...] + q*q) - (grid_v_k + v_k*v_k))/(2*q - 2*v_k)
        z_k = np.take_along_axis(z, k[None, :], axis=0) # Get the value of z at corresponding k indices
        mask = (s <= z_k)
        while mask.any():
            # Update k values of those 1D-arrays whose value is now less than of equal to z
            # Loop won't terminate till all the 1D arrays satisfies the condition
            k[np.squeeze(mask, axis=0)] -= 1
            v_k = np.take_along_axis(v, k[None, :], axis=0).astype(np.int32)
            grid_v_k = np.take_along_axis(grid, v_k, axis=0)
            s = ((grid[q,...] + q*q) - (grid_v_k + v_k*v_k))/(2*q - 2*v_k)
            z_k = np.take_along_axis(z, k[None, :], axis=0)
            mask = (s <= z_k)
        k = k+1 
        np.put_along_axis(v, k[None, :], q, axis=0)
        np.put_along_axis(z, k[None, :], s, axis=0)
        np.put_along_axis(z, (k+1)[None, :], q, axis=0)
    
    k[...] = 0
    
    ans = np.zeros(grid.shape)
    for q in range(N):
        k_1 = k+1
        z_k_1 = np.take_along_axis(z, k_1[None, :], axis=0)
        mask = (z_k_1 < q)
        while mask.any():
            k[np.squeeze(mask, axis=0)] += 1
            k_1 = k+1
            z_k_1 = np.take_along_axis(z, k_1[None, :], axis=0)
            mask = (z_k_1 < q)
        v_k = np.take_along_axis(v, k[None, :], axis=0).astype(np.int32)
        ans[q,...] = (q - v_k)*(q-v_k) + np.take_along_axis(grid, v_k, axis=0)
        
    return ans

def esdf(M, N, obstacle_list):
    """
    :param M: Row number
    :param N: Column number
    :param obstacle_list: Obstacle list
    :return: An array. The value of each cell means the closest distance to the obstacle
    """
    grid = np.ones((M,N))*INF
    obstacle_list = np.array(obstacle_list)
    grid[tuple(obstacle_list.T)] = 0
    df1 = esdf_1d(grid) # Gets the 1D-array ESDF along the columns 
    df = np.ones(grid.shape)*INF
    grid_x = np.ones(grid.shape)*np.array(range(N))
    for q in range(N):
        dist_x_2 = (grid_x - q)*(grid_x-q) # Get distance square along the axis
        df_q = df1 + dist_x_2 # Get square of total overhead distance to obstacle
        df[...,q] = np.amin(df_q, axis=-1)
    
    df = np.sqrt(df)
    return df

if __name__ == '__main__':
    st = time.time()
    for _ in range(int(2e4)):
        assert np.array_equal(esdf(M=3, N=3, obstacle_list=[[0, 1], [2, 2]]), res_1)
        assert np.array_equal(esdf(M=4, N=5, obstacle_list=[[0, 1], [2, 2], [3, 1]]), res_2)

    et = time.time()
    print(et-st)

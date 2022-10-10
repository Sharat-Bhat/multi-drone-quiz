import time
from result import *
import numpy as np

INF = 1000000000

def esdf_1d(grid):
    """Calculates the distance to the closest obstacle along the first axis

    Args:
        grid (float64): Holds the upper bound on the nearest obstacle distance along the axis
    """
    N = grid.shape[0]
    # print(N)
    k = np.zeros(grid.shape[1:]).astype(np.int32) # Index of rightmost element in lower envelope
    v = np.ones(grid.shape)*INF # Locations of parabolas in lower envelope
    v[0,...] = 0
    z = np.zeros((N+1,) + grid.shape[1:]) # Location of boundaries between parabolas
    z[0,...] = -INF
    z[1,...] = INF
    
    for q in range(1, N):
        # print(q)
        v_k = np.take_along_axis(v, k[None, :], axis=0).astype(np.int32)
        # print(v_k)
        grid_v_k = np.take_along_axis(grid, v_k, axis=0)
        # print(grid_v_k)
        s = ((grid[q,...] + q*q) - (grid_v_k + v_k*v_k))/(2*q - 2*v_k)
        # print(s)
        z_k = np.take_along_axis(z, k[None, :], axis=0)
        # print(z_k)
        mask = (s <= z_k)
        # print(mask)
        # print(k)
        while mask.any() and np.amin(k):
            k[np.squeeze(mask, axis=0)] -= 1
            # k = np.where(np.squeeze(mask, axis=0), k-1, k)
            v_k = np.take_along_axis(v, k[None, :], axis=0).astype(np.int32)
            # print(v_k)
            grid_v_k = np.take_along_axis(grid, v_k, axis=0)
            # print(grid_v_k)
            s = ((grid[q,...] + q*q) - (grid_v_k + v_k*v_k))/(2*q - 2*v_k)
            # print(s)
            z_k = np.take_along_axis(z, k[None, :], axis=0)
            # print(z_k)
            mask = (s <= z_k)
            # print(mask)
            # print(k)
            
        
        k = k+1 
        np.put_along_axis(v, k[None, :], q, axis=0)
        np.put_along_axis(z, k[None, :], s, axis=0)
        np.put_along_axis(z, (k+1)[None, :], q, axis=0)
    
    k[...] = 0
    
    ans = np.zeros(grid.shape)
    for q in range(N):
        # print("Q", q)
        k_1 = k+1
        z_k_1 = np.take_along_axis(z, k_1[None, :], axis=0)
        # print(z_k_1)
        mask = (z_k_1 < q)
        # print(mask)
        while mask.any():
            k[np.squeeze(mask, axis=0)] += 1
            # k = np.where(np.squeeze(mask, axis=0), k-1, k)
            k_1 = k+1
            z_k_1 = np.take_along_axis(z, k_1[None, :], axis=0)
            # print(z_k_1)
            mask = (z_k_1 < q)
            # print(mask)
        v_k = np.take_along_axis(v, k[None, :], axis=0).astype(np.int32)
        ans[q,...] = (q - v_k)*(q-v_k) + np.take_along_axis(grid, v_k, axis=0)
        
    # print(ans)
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
    # print(obstacle_list)
    grid[tuple(obstacle_list.T)] = 0
    # print(grid)
    df1 = esdf_1d(grid)
    df = np.ones(grid.shape)*INF
    # print(df1)
    grid_x = np.ones(grid.shape)*np.array(range(N))
    # print(grid_x)
    for q in range(N):
        dist_x_2 = (grid_x - q)*(grid_x-q)
        df_q = df1 + dist_x_2
        # print(df_q)
        df[...,q] = np.amin(df_q, axis=-1)
    
    df = np.sqrt(df)
    # print(df)
    return df

if __name__ == '__main__':
    st = time.time()
    for _ in range(int(2e4)):
        assert np.array_equal(esdf(M=3, N=3, obstacle_list=[[0, 1], [2, 2]]), res_1)
        assert np.array_equal(esdf(M=4, N=5, obstacle_list=[[0, 1], [2, 2], [3, 1]]), res_2)

    et = time.time()
    print(et-st)

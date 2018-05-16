# Copyright (c) 2018 Peter Limkilde Svendsen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, lsqr, lsmr

def laplacian_gapfill(input_grid):
    num_rows, num_cols = input_grid.shape
    
    has_data = np.isfinite(input_grid)
    
    if np.sum(~has_data) == 0:
        # No missing data, nothing to solve for
        return input_grid
    elif np.sum(has_data) == 0:
        # All data missing, return all-NaN grid
        return np.full((num_rows,num_cols), np.nan)
    else:
        # General case
        num_pixels_total = num_rows*num_cols
        num_pixels_data = np.sum(has_data)
        num_pixels_nodata = num_pixels_total - num_pixels_data
        
        has_neighbor_above = np.ones_like(input_grid, dtype=np.bool)
        has_neighbor_left  = np.ones_like(input_grid, dtype=np.bool)
        has_neighbor_right = np.ones_like(input_grid, dtype=np.bool)
        has_neighbor_below = np.ones_like(input_grid, dtype=np.bool)
        
        has_neighbor_above[0,:]  = False
        has_neighbor_left[:,0]   = False
        has_neighbor_right[:,-1] = False
        has_neighbor_below[-1,:] = False
        
        num_neighbors = has_neighbor_above.astype(np.int) + has_neighbor_left.astype(np.int) + has_neighbor_right.astype(np.int) + has_neighbor_below.astype(np.int)
        
        rowgrid, colgrid = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing='ij')
        
        pixels_total_range = np.arange(num_pixels_total)
        
        # Build discrete Laplacian matrix, starting with center weights...
        center_matrix = coo_matrix((-num_neighbors.ravel().astype(np.float), (pixels_total_range, pixels_total_range)), shape=(num_pixels_total, num_pixels_total))
        
        # ... add neighbors above (num_cols pixels prior)...
        neighbor_matrix = coo_matrix((np.ones((np.sum(has_neighbor_above),)), (pixels_total_range[has_neighbor_above.ravel()].ravel(), pixels_total_range[has_neighbor_above.ravel()].ravel()-num_cols)), shape=(num_pixels_total, num_pixels_total))
        
        # ... add neighbors left (1 pixel prior)...
        neighbor_matrix += coo_matrix((np.ones((np.sum(has_neighbor_left),)), (pixels_total_range[has_neighbor_left.ravel()].ravel(), pixels_total_range[has_neighbor_left.ravel()].ravel()-1)), shape=(num_pixels_total, num_pixels_total))
        
        # ... add neighbors right (1 pixel after)...
        neighbor_matrix += coo_matrix((np.ones((np.sum(has_neighbor_right),)), (pixels_total_range[has_neighbor_right.ravel()].ravel(), pixels_total_range[has_neighbor_right.ravel()].ravel()+1)), shape=(num_pixels_total, num_pixels_total))
        
        # ... add neighbors below (num_cols pixels after).
        neighbor_matrix += coo_matrix((np.ones((np.sum(has_neighbor_below),)), (pixels_total_range[has_neighbor_below.ravel()].ravel(), pixels_total_range[has_neighbor_below.ravel()].ravel()+num_cols)), shape=(num_pixels_total, num_pixels_total))
        
        # Merge center point and neighbor weights
        system_matrix = center_matrix + neighbor_matrix
        
        # Compute contributions from pixels that have data...
        data_contribs = system_matrix[:,has_data.ravel()].dot(input_grid.ravel()[has_data.ravel()])
        
        # ... and substitute into the right-hand side.
        rhs = -data_contribs
        
        # Select the nodata pixels for solving
        system_matrix = system_matrix[:, ~(has_data.ravel())]
        
        # Solve for the gap areas
        nodata_solution = lsmr(system_matrix, rhs)[0]
        
        output_grid_flat = input_grid.ravel().copy()
        
        output_grid_flat[~has_data.ravel()] = nodata_solution
        
        output_grid = output_grid_flat.reshape(num_rows, num_cols)
        
        return output_grid

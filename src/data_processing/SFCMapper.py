'''
SFCMapper.py
Updated: 2/6/18

This script contains methods to generate space filling curves and uses them to
map high dimensional data into lower dimensions.

'''
import numpy as np

class SFCMapper(object):
    """
    SFCMapper object is used to generate 3D and 2D space filling curves and map
    the traveseral of 3D curves to 2D curves. This allows 3D discrete spaces to
    be reduced to 2D discrete spaces.

    """

    def __init__(self, size_3d):
        '''
        Method generates 2D and 3D space filling curves used for mapping 3D to
        2D.

        Param:
            size_3d - int ; length of 3D space

        '''
        # Set size variables
        self.size_3d = size_3d
        self.size_2d = np.sqrt(self.size_3d**3)
        if self.size_2d % 1.0 != 0:
            print("Error: 3D Space not mappable to 2D with Hilbert Curve.");exit()
        else: self.size_2d = int(self.size_2d)

        # Generate Curves
        print("Generating Space-Filling Curves...")
        self.curve_3d = self.__hilbert_3d(int(np.log2(self.size)))
        self.curve_2d = self.__hilbert_2d(int(np.log2(self.size_2d)))

    def map_3d_to_2d(array_3d):
        '''
        Method proceses 3D array and encodes into 2D using SFCs.

        Param:
            array_3d - np.array

        Return:
            array_2d - np.array

        '''
        s = int(np.cbrt(len(curve_3d)))
        array_3d = np.zeros([s,s,s])
        for i in range(s**3):
            c2d = self.curve_2d[i]
            c3d = self.curve_3d[i]
            array_2d[c2d[0], c2d[1]] = array_3d[c3d[0], c3d[1], c3d[2]]

        return array_2d

    def __hilbert_3d(self, order):
        '''
        Method generates 3D hilbert curve of desired order.

        Param:
            order - int ; order of curve

        Returns:
            np.array ; list of (x, y, z) coordinates of curve

        '''

        def gen_3d(order, x, y, z, xi, xj, xk, yi, yj, yk, zi, zj, zk, array):
            if order == 0:
                xx = x + (xi + yi + zi)/3
                yy = y + (xj + yj + zj)/3
                zz = z + (xk + yk + zk)/3
                array.append((xx, yy, zz))
            else:
                gen_3d(order-1, x, y, z, yi/2, yj/2, yk/2, zi/2, zj/2, zk/2, xi/2, xj/2, xk/2, array)

                gen_3d(order-1, x + xi/2, y + xj/2, z + xk/2,  zi/2, zj/2, zk/2, xi/2, xj/2, xk/2,
                           yi/2, yj/2, yk/2, array)
                gen_3d(order-1, x + xi/2 + yi/2, y + xj/2 + yj/2, z + xk/2 + yk/2, zi/2, zj/2, zk/2,
                           xi/2, xj/2, xk/2, yi/2, yj/2, yk/2, array)
                gen_3d(order-1, x + xi/2 + yi, y + xj/2+ yj, z + xk/2 + yk, -xi/2, -xj/2, -xk/2, -yi/2,
                           -yj/2, -yk/2, zi/2, zj/2, zk/2, array)
                gen_3d(order-1, x + xi/2 + yi + zi/2, y + xj/2 + yj + zj/2, z + xk/2 + yk +zk/2, -xi/2,
                           -xj/2, -xk/2, -yi/2, -yj/2, -yk/2, zi/2, zj/2, zk/2, array)
                gen_3d(order-1, x + xi/2 + yi + zi, y + xj/2 + yj + zj, z + xk/2 + yk + zk, -zi/2, -zj/2,
                           -zk/2, xi/2, xj/2, xk/2, -yi/2, -yj/2, -yk/2, array)
                gen_3d(order-1, x + xi/2 + yi/2 + zi, y + xj/2 + yj/2 + zj , z + xk/2 + yk/2 + zk, -zi/2,
                           -zj/2, -zk/2, xi/2, xj/2, xk/2, -yi/2, -yj/2, -yk/2, array)
                gen_3d(order-1, x + xi/2 + zi, y + xj/2 + zj, z + xk/2 + zk, yi/2, yj/2, yk/2, -zi/2, -zj/2,
                           -zk/2, -xi/2, -xj/2, -xk/2, array)

        n = pow(2, order)
        hilbert_curve = []
        gen_3d(order, 0, 0, 0, n, 0, 0, 0, n, 0, 0, 0, n, hilbert_curve)

        return np.array(hilbert_curve).astype('int')

    def __hilbert_2d(self, order):
        '''
        Method generates 2D hilbert curve of desired order.

        Param:
            order - int ; order of curve

        Returns:
            np.array ; list of (x, y) coordinates of curve

        '''
        def gen_2d(order, x, y, xi, xj, yi, yj, array):
            if order == 0:
                xx = x + (xi + yi)/2
                yy = y + (xj + yj)/2
                array.append((xx, yy))
            else:
                gen_2d(order-1, x, y, yi/2, yj/2, xi/2, xj/2, array)
                gen_2d(order-1, x + xi/2, y + xj/2, xi/2, xj/2, yi/2, yj/2, array)
                gen_2d(order-1, x + xi/2 + yi/2, y + xj/2 + yj/2, xi/2, xj/2, yi/2, yj/2, array)
                gen_2d(order-1, x + xi/2 + yi, y + xj/2 + yj, -yi/2,-yj/2,-xi/2,-xj/2, array)

        n = pow(2, order)
        hilbert_curve = []
        gen_2d(order, 0, 0, n, 0, 0, n, hilbert_curve)

        return np.array(hilbert_curve).astype('int')

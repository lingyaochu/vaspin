from json import load,loads
import os
import re

import numpy as np
import numpy.typing as npt


def clean(target_dir: str) -> None:
    """
    若targetdir不存在，那么创建它

    Args:
        target_dir: 目标文件夹的路径

    Returns:
        None
    """
    if os.path.exists(target_dir):
        pass
    else:
        os.makedirs(target_dir)

    return None


class poscar:
    """
    要配合linux上的脚本导出的pos.json 使用
    处理vasp中的POSCAR的程序
    """

    def __init__(self, posfile: str):

        __massdict = {
            'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.812, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974, 'S': 32.066,
            'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942,
            'Cr': 51.996,
            'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.382, 'Ga': 69.723,
            'Ge': 72.641,
            'As': 74.922, 'Se': 78.972, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.621, 'Y': 88.906,
            'Zr': 91.224,
            'Nb': 92.906, 'Mo': 95.951, 'Tc': 98.907, 'Ru': 101.072, 'Rh': 102.906, 'Pd': 106.421, 'Ag': 107.868,
            'Cd': 112.414, 'In': 114.818, 'Sn': 118.711, 'Sb': 121.760, 'Te': 127.603, 'I': 126.904, 'Xe': 131.294,
            'Cs': 132.905, 'Ba': 137.328, 'La': 138.905, 'Ce': 140.116, 'Pr': 140.908, 'Nd': 144.242, 'Pm': 144.920,
            'Sm': 150.362, 'Eu': 151.964, 'Gd': 157.253, 'Tb': 158.925, 'Dy': 162.500, 'Ho': 164.930, 'Er': 167.259,
            'Tm': 168.934, 'Yb': 173.055, 'Lu': 174.967, 'Hf': 178.492, 'Ta': 180.948, 'W': 183.841, 'Re': 186.207,
            'Os': 190.233, 'Ir': 192.217, 'Pt': 195.085, 'Au': 196.967, 'Hg': 200.592, 'Tl': 204.383, 'Pb': 207.210,
            'Bi': 208.980, 'Po': 208.982, 'At': 209.987, 'Rn': 222.018, 'Fr': 223.020, 'Ra': 226.025, 'Ac': 227.028,
            'Th': 232.038, 'Pa': 231.036, 'U': 238.029, 'Np': 237.048, 'Pu': 239.064, 'Am': 243.061, 'Cm': 247.070,
            'Bk': 247.070, 'Cf': 251.080, 'Es': 252.083, 'Fm': 257.059, 'Md': 258.098, 'No': 259.101, 'Lr': 262.110,
            'Rf': 267.122, 'Db': 268.126, 'Sg': 269.129, 'Bh': 274.144, 'Hs': 277.152, 'Mt': 278, 'others': 281
        }

        try:
            with open(posfile, 'r') as f:
                __data = load(f)
        except:
            __data = loads(poscar.poscar_to_json(posfile))

        self.lattice = self.lattice(__data['coe'], __data['lattice'])

        self.abc = {'a': np.linalg.norm(self.lattice[0]),
                    'b': np.linalg.norm(self.lattice[1]),
                    'c': np.linalg.norm(self.lattice[2])}
        self.volume = self.volume(self.lattice)

        self.atom = np.array(__data['species'])
        self.number = np.array(__data['number'])

        self.species = []
        for i in range(len(self.atom)):
            self.species += [self.atom[i]] * self.number[i]
        self.species = np.array(self.species)

        self.mass = []
        append = self.mass.append
        for i in self.species:
            append(__massdict[i])
        self.mass = np.array(self.mass)

        if __data['coortype'] == 'Direct':
            self.coor_frac = np.array(__data['coordinate'])
            self.coor_cate = self.frac_to_cate(self.coor_frac)

        elif __data['coortype'] == 'Cartesian':
            self.coor_cate = np.array(__data['coordinate'])
            self.coor_frac = self.cate_to_frac(self.coor_cate)

    @staticmethod
    def lattice(coe: float, lat: list[list[float]]) -> npt.NDArray[np.float64]:
        """
        根据输入的coe和lat，计算晶格矢量

        Args:
            coe: 放缩系数
            lat: 未放缩的晶格矢量

        Returns:
            放缩后的晶格矢量

        Examples:
            >>> poscar.lattice(1.0, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 1.0]])
            array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 1.0]])

            >>> poscar.lattice(2.0, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 1.0]])
            array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0, 0, 2.0]])
        """
        lat = np.array(lat)

        return coe * lat

    @staticmethod
    def volume(lattice: list[list[float]] | npt.NDArray) -> np.float64:
        """
        对输入的晶格矢量，通过行列式的方法计算晶胞体积

        Args:
            lattice: 放缩之后的晶格矢量

        Returns:
            晶胞体积

        Examples:
            >>> poscar.volume([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 1.0]])
            1.0

            >>> poscar.volume(array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0, 0, 2.0]]))
            8.0

        """
        lattice = np.array(lattice)
        vol = np.linalg.det(lattice)

        return vol

    def frac_to_cate(self, frac: npt.NDArray, lattice: npt.NDArray = np.array([])) -> npt.NDArray[np.float64]:
        """
        将直接坐标转化成笛卡尔坐标

        Args:
            frac: 原子的分数坐标
            lattice: 晶格矢量

        Returns:
            原子的笛卡尔坐标

        Examples:
            >>> _lattice = np.array([[10,0,0],[0,10,0],[0,0,10]])
            >>> _frac = np.array([[0.5,0.5,0.5],[0.1,0.1,0.1]])
            >>> poscar.frac_to_cate(_lattice, _frac)
            array([[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]])

        """
        if len(lattice) == 0:
            lattice = self.lattice.copy()
        else:
            lattice = np.array(lattice)
        cate = frac @ lattice

        return cate

    def cate_to_frac(self, cate: npt.NDArray, lattice: npt.NDArray = np.array([])) -> npt.NDArray[np.float64]:
        """
        将笛卡尔坐标转化成直接坐标

        Args:
            cate: 原子的笛卡尔坐标
            lattice: 放缩后的晶格矢量

        Returns:
            原子的分数坐标

        Examples:
            >>> _lattice = np.array([[10,0,0],[0,10,0],[0,0,10]])
            >>> _cate = np.array([[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]])
            >>> poscar.cate_to_frac(_cate, _lattice)
            array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])

        """
        if len(lattice) == 0:
            lattice = self.lattice.copy()
        else:
            lattice = np.array(lattice)
        frac = cate @ np.linalg.inv(lattice)

        return frac

    @staticmethod
    def _lattice_to_str(lattice: npt.NDArray) -> str:
        """
        将晶格矢量转化成字符串

        Args:
            lattice: 晶格矢量

        Returns:
            转换成字符串形式的晶格矢量

        Examples:
            >>> _lattice = np.array([[10,0,0],[0,10,0],[0,0,10]])
            >>> poscar._lattice_to_str(_lattice)
            '   10.0000000000000000    0.0000000000000000    0.0000000000000000
               0.0000000000000000   10.0000000000000000    0.0000000000000000
               0.0000000000000000    0.0000000000000000   10.0000000000000000'

        """
        fullstr = ""

        for i in range(3):
            rowstr = ""
            for j in range(3):
                rowstr = rowstr + "    " + "{:.16f}".format(lattice[i][j])
            fullstr = fullstr + rowstr + "\n"
        return fullstr

    @staticmethod
    def _atoms_and_numbers(species: npt.NDArray[np.str_]) -> tuple[list[str], list[int]]:

        """
        将原子种类转换成可以写入POSCAR的格式

        Args:
            species: 原子种类列表

        Returns:
            atoms: 元素种类
            numbers: 元素数量

        Examples:
            >>> _species = np.array(['Ga', 'Ga', 'Ga', 'O', 'O', 'O'])
            >>> poscar._atoms_and_numbers(_species)
            (['Ga', 'O'], [3, 3])
        """

        atoms, numbers = poscar._unique_with_order_and_counts(species)

        return atoms, numbers

    def _specie_to_str(self, species: list[str] = None) -> str:
        """
        将原子种类转换成字符串形式

        Args:
            species: 原子种类列表

        Returns:
            字符串形式的原子种类

        Examples:
            >>> _species = ['Ga', 'Ga', 'Ga', 'O', 'O', 'O']
            >>> self._specie_to_str(_species)
            '   Ga   O
               3    3'

        """
        if species is None:
            atoms = self.atom.copy()
            numbers = self.number.copy()
        else:
            atoms, numbers = self._atoms_and_numbers(species)
        fullstr = ""
        for i in range(len(atoms)):
            fullstr = fullstr + "   " + atoms[i]
        fullstr = fullstr + "\n"
        for i in range(len(numbers)):
            fullstr = fullstr + "    " + str(numbers[i])

        return fullstr

    @staticmethod
    def _coordinate_to_str(coordinate: npt.NDArray) -> str:
        """
        将坐标转换成字符串形式

        Args:
            coordinate: 原子的坐标列表

        Returns:
            字符串形式的原子坐标

        Examples:
            >>> _coordinate = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
            >>> poscar._coordinate_to_str(_coordinate)
            '   0.5000000000000000    0.5000000000000000    0.5000000000000000
               0.1000000000000000    0.1000000000000000    0.1000000000000000'

        """
        fullstr = ""
        for i in range(len(coordinate)):
            rowstr = ""
            for j in range(3):
                rowstr = rowstr + "    " + "{:.16f}".format(coordinate[i][j])
            fullstr = fullstr + rowstr + "\n"

        return fullstr

    def write_poscar(self, lattice: list[list[float]] = None, species: list[str] = None,
                     coor_frac: list[list[float]] = None, directory: str = ".",
                     comment: str = "generated by mother python",
                     name: str = "POSCAR") -> None:
        """
        将数据写入到POSCAR文件中

        Args:
            lattice: 晶格数据，默认使用对象本身的数据
            species: 原子种类数据，默认使用对象本身的数据
            coor_frac: 分数坐标数据，默认使用对象本身的数据
            directory: 写入文件的路径，默认为当前路径
            comment: POSCAR文件中第一行的描述信息
            name: POSCAR文件的名称，默认为POSCAR

        Returns:
            None

        Examples:
            >>> self.write_poscar()

        """
        if lattice is None:
            lattice = self.lattice.copy()

        if coor_frac is None:
            coor_frac = self.coor_frac.copy()

        if species is None:
            species = self.species.copy()

        sort = np.argsort(species,kind='stable')
        coor_frac = coor_frac[sort]
        species = species[sort]

        lattice_str = self._lattice_to_str(lattice)
        coor_str = self._coordinate_to_str(coor_frac)
        species_str = self._specie_to_str(species)
        clean(directory)
        with open(directory + f"/{name}", "w") as f:
            f.write(comment + "\n")
            f.write(" 1.0\n")
            f.write(lattice_str)
            f.write(species_str + "\n")
            f.write("Direct" + "\n")
            f.write(coor_str)
        if not np.array_equal(self.species,species):
            print("Warning: The order of the elements in the POSCAR file may be changed.")
        return None
    
    @staticmethod
    def poscar_to_json(filepath: str) -> str:
        """
        将POSCAR读取成json格式的字符串
        """
        with open(filepath,'r') as f:
            f.readline()
            data_str = "{\n"
            coe_lattice = f.readline().strip()
            
            data_str = data_str + "\"coe\": " + coe_lattice + ",\n"
            data_str = data_str + "\"lattice\": \n"
            
            lattice_a = re.split(r'\s{1,}',f.readline().strip())
            lattice_b = re.split(r'\s{1,}',f.readline().strip())
            lattice_c = re.split(r'\s{1,}',f.readline().strip())
            
            data_str = data_str + "[[" + lattice_a[0] + "," + lattice_a[1] + "," + lattice_a[2] + "],\n"
            data_str = data_str + " [" + lattice_b[0] + "," + lattice_b[1] + "," + lattice_b[2] + "],\n"
            data_str = data_str + " [" + lattice_c[0] + "," + lattice_c[1] + "," + lattice_c[2] + "]],\n"
            
            # for i in range(3):
            #     lattice_a[i] = float(lattice_a[i])
            #     lattice_b[i] = float(lattice_b[i])
            #     lattice_c[i] = float(lattice_c[i])
                
            atom_type = re.split(r'\s{1,}',f.readline().strip())
            for i in range(len(atom_type)):
                # 为了适配vasp6.4版本之后的POSCAR
                atom_type[i] = re.split(r'/', atom_type[i])[0]

            atom_numb = re.split(r'\s{1,}',f.readline().strip())
            atom_numb_str = "\"number\": " + "["
            data_str = data_str + "\"species\": ["
            atoms = 0
            for i in range(len(atom_numb)):
                data_str = data_str + "\"" + atom_type[i] + "\"" + ","
                atom_numb_str = atom_numb_str + atom_numb[i] + ","
                atom_numb[i] = int(atom_numb[i])
                atoms = atoms + atom_numb[i]
            
            data_str = data_str.rstrip(',')
            data_str = data_str + "],\n"
            atom_numb_str = atom_numb_str.rstrip(',') + "],"
            data_str = data_str + atom_numb_str + "\n"
            
            type_coor = f.readline().strip()
            data_str = data_str + "\"coortype\": " + "\""+type_coor+"\",\n"
            coor_str = "["
            for i in range(atoms):
                coor_x, coor_y, coor_z = re.split(r'\s{1,}',f.readline().strip())
                coor_str = coor_str + "[" + coor_x + "," + coor_y + "," + coor_z + "],\n"
            
            coor_str = coor_str.rsplit('\n',1)[0].rstrip(',') + "]\n"
            
            data_str = data_str + "\"coordinate\": \n" + coor_str + '}'
        return data_str


    def random_disp(self, magnitude: float = 0.1, method="cate") -> npt.NDArray[np.float64]:
        """
        对结构中的原子施加随机扰动

        Args:
            magnitude: 原子扰动的最大距离，默认为0.1 A
            method: 扰动的方式 cate: 沿x,y,z三个方向进行随机大小的扰动；sphere: 随机长度，随机方向的扰动

        Returns:
            扰动的方向矢量

        Examples:
            >>> self.random_disp(0.1)
            [[0.057737,0.01256,0.02355],[0.011204,0.013489,0.022355],...]
        """
        assert method in ["cate", "sphere"], "method must be cate or sphere"
        assert magnitude > 0, "magnitude must be greater than 0"

        if method == "cate":
            magnitude_one_direction = magnitude / np.sqrt(3)

            disp_random = magnitude_one_direction * np.random.random(self.coor_cate.shape)

        else:
            phi = np.random.uniform(0, 2 * np.pi, size = self.coor_cate.shape[0])
            theta = np.arccos(np.random.uniform(-1, 1, size = self.coor_cate.shape[0]))
            r = np.power(np.random.uniform(0, magnitude**3, size = self.coor_cate.shape[0]),1/3)
            disp_random = np.zeros(self.coor_cate.shape)

            r_sin_theta = r * np.sin(theta)
            disp_random[:,0] = r_sin_theta * np.cos(phi)
            disp_random[:,1] = r_sin_theta * np.sin(phi)
            disp_random[:,2] = r * np.cos(theta)

        return disp_random

    def distance(self, atom_index: int = None, frac_coor: npt.NDArray | list[float] = None, detail: bool = False) -> npt.NDArray | tuple[npt.NDArray, tuple[np.int64, str, np.float64]]:
        """
        计算结构中某个原子或者某个坐标和其他原子之间的距离, 并返回相应的信息
        Args:
            atom_index: 目标原子在结构中的序号
            frac_coor: 某个具体的坐标
            detail: 是否返回详细的距离信息

        Returns:
            distance_list: 按照原子顺序的距离列表
            按照距离排序的原子序号，元素符号，距离

        """

        # 将目标原子摆在结构的中心
        if atom_index is not None:
            center_atom_coor_frac = self.coor_frac[atom_index-1].copy()
        else:
            center_atom_coor_frac = np.array(frac_coor)

        cell_center = np.array([0.5, 0.5, 0.5])
        arrow_to_middle = cell_center - center_atom_coor_frac

        # 移动所有原子，保证目标原子摆在中心
        coor_frac_moved = self.coor_frac + arrow_to_middle
        for i in range(coor_frac_moved.shape[0]):
            coor_frac_moved[i] = self.pull_coor_frac(coor_frac_moved[i])

        # 计算距离
        distance_arrow = self.frac_to_cate(coor_frac_moved - cell_center)
        distance_list = np.linalg.norm(distance_arrow, axis=1)

        if not detail:
            return distance_list
        else:
            sorted_index = np.argsort(distance_list,kind='stable')
            distance_sorted = distance_list[sorted_index]
            species_sorted = self.species[sorted_index]
            return distance_list, tuple(zip(sorted_index, species_sorted, distance_sorted))
    
    def find_local(self, atom_index: int = None, frac_coor: npt.NDArray | list[float] = None, dmax: float = 0.2) -> list[int]:
        """
        找出对应原子为中心，限定半径下的局域结构

        Args:
            atom_index: 中心原子的序号
            frac_coor: 中心原子的分数坐标
            dmax: 搜索半径，默认为0.2
        
        Returns:
            局域结构的原子序号列表
        """
        distance_tuple = self.distance(atom_index, frac_coor, detail=True)[1]
        local_stru_index = []
        for i in distance_tuple:
            if i[2] <= dmax:
                local_stru_index.append(i[0])
        
        return local_stru_index
    
    def find_neighbor(self, inputatom: int | float, dmax: float = 0.5) -> dict[str, str | float]:
        """
        找出对应原子的最近邻原子

        Args:
            inputatom: 中心原子的序号或者元素符号
            dmax: 最近邻原子的壳层范围，默认为0.7

        Returns:
            最近邻原子的原子列表，分数坐标列表，以及将中心原子移动到中心后的分数坐标列表

        Examples:
            >>> self.find_neighbor(1,0.9)

            >>>self.find_neighbor("Ga",0.9)
        """
        atomnumber = 0
        # 判断inputatom是否是一个整型
        if isinstance(inputatom, int):
            atomnumber = inputatom - 1

        # 判断inputatom是否是一个字符串
        elif isinstance(inputatom, str):
            atomnumber = self.species.index(inputatom)

        distance = {}

        # 首先把中心原子移动到胞的中心
        center_atom_coor_frac = self.coor_frac[atomnumber].copy()
        center_coor_frac = np.array([0.5, 0.5, 0.5])
        disp_frac = np.array(center_coor_frac) - center_atom_coor_frac
        all_disped_frac = []
        for i in range(len(self.species)):
            all_disped_frac.append(self.pull_coor_frac(self.coor_frac[i] + disp_frac))
        all_disped_frac = np.array(all_disped_frac)

        all_disped_cate = self.frac_to_cate(self.lattice, all_disped_frac)

        for i in range(len(self.species)):
            if i == atomnumber:
                continue
            else:
                distance[i] = np.linalg.norm(all_disped_cate[i] - all_disped_cate[atomnumber])
        min_distance = min(list(distance.values()))
        max_distance = min(list(distance.values())) + dmax

        neighbor_index = []
        for i in distance.keys():
            if min_distance <= distance[i] <= max_distance:
                neighbor_index.append(i)

        # 将中心原子添加到得到的最近邻原子列表中
        neighbor_index.append(atomnumber)

        neighbor_species = self.species[neighbor_index]
        neighbor_coor_frac = self.coor_frac[neighbor_index]
        neighbor_disped_frac = all_disped_frac[neighbor_index]

        neighbor_origin_frac = []
        for i in range(len(neighbor_coor_frac)):
            neighbor_origin_frac.append(neighbor_coor_frac[i] - neighbor_coor_frac[-1])

        results = {"species": neighbor_species,
                   "coor_frac": neighbor_coor_frac,
                   "coor_frac_centered": neighbor_disped_frac,
                   "coor_frac_zero": neighbor_origin_frac,
                   "coor_cate": self.frac_to_cate(self.lattice, neighbor_coor_frac),
                   "coor_cate_centered": self.frac_to_cate(self.lattice, neighbor_disped_frac),
                   "coor_cate_zero": self.frac_to_cate(self.lattice, neighbor_origin_frac)}

        return results

    @staticmethod
    def _unique_with_order_and_counts(inputlist: list | npt.NDArray) -> tuple[list, list]:
        """
        在顺序不变的情况下，返回列表中的唯一值，以及每个唯一值出现的次数。和np.unique()相比，这个函数可以保证顺序不变

        Parametes:
            @inputlist: 输入的列表

        Returns:
            @return: 唯一值列表，以及次数列表

        Example:
            >>> poscar._unique_with_order_and_counts(['O','O','O','O','Ga'])
            (['O', 'Ga'], [4, 1])
        """

        inputlist = np.array(inputlist)
        _, idx, counts = np.unique(inputlist, return_index=True, return_counts=True)
        sorted_index = np.argsort(idx)

        return inputlist[idx[sorted_index]].tolist(), counts[sorted_index].tolist()

    @staticmethod
    def pull_coor_frac(coor_frac: list[float] | npt.NDArray) -> npt.NDArray[np.float64]:
        """
        将超过第一晶胞范围的原子坐标转换回第一晶胞

        Args:
            coor_frac: 分数坐标

        Returns:
            转换后的分数坐标

        Examples:
            >>> poscar.pull_coor_frac([0.1,0.2,0.3])
            [0.1,0.2,0.3]

            >>> poscar.pull_coor_frac([1.1,0.2,0.3])
            [0.1,0.2,0.3]

            >>> poscar.pull_coor_frac([-0.1,0.2,0.3])
            [0.9,0.2,0.3]
        """
        coor_frac = np.array(coor_frac)
        coor_frac = coor_frac - np.floor(coor_frac)

        return coor_frac

    @staticmethod
    def find_rotation(arrow1: list[float] | npt.NDArray, arrow2: list[float] | npt.NDArray) -> npt.NDArray[np.float64]:
        """
        根据一般三维空间中的一般转动矩阵形式，找到将arrow1转到arrow2的旋转矩阵

        Args:
            arrow1: 转动前的三维矢量
            arrow2: 转动后的三维矢量

        Returns:
            三维转动矩阵

        Examples:
            >>> poscar.find_rotation([1,0,0],[0,1,0])
            array([[ 0, -1, 0],
                   [ 1, 0, 0],
                   [ 0, 0, 1]])
        """
        rotation_axis = np.cross(arrow1, arrow2) / np.linalg.norm(np.cross(arrow1, arrow2))
        theta = np.arccos(np.dot(arrow1, arrow2) / np.linalg.norm(arrow1) / np.linalg.norm(arrow2))
        if rotation_axis[2] < 0:
            rotation_axis = -rotation_axis
            theta = -theta
        x, y, z = rotation_axis.copy()

        # 有转轴和转角，我们可以自行构造旋转矩阵
        _matrix = np.zeros((3, 3))
        _matrix[0, 0] = np.cos(theta) + x ** 2 * (1 - np.cos(theta))
        _matrix[0, 1] = x * y * (1 - np.cos(theta)) - z * np.sin(theta)
        _matrix[0, 2] = x * z * (1 - np.cos(theta)) + y * np.sin(theta)
        _matrix[1, 0] = y * x * (1 - np.cos(theta)) + z * np.sin(theta)
        _matrix[1, 1] = np.cos(theta) + y ** 2 * (1 - np.cos(theta))
        _matrix[1, 2] = y * z * (1 - np.cos(theta)) - x * np.sin(theta)
        _matrix[2, 0] = z * x * (1 - np.cos(theta)) - y * np.sin(theta)
        _matrix[2, 1] = z * y * (1 - np.cos(theta)) + x * np.sin(theta)
        _matrix[2, 2] = np.cos(theta) + z ** 2 * (1 - np.cos(theta))

        return _matrix

    def lattice_rotation(self, matrix_rot: npt.NDArray):
        """
        转动晶格矢量

        Args:
            matrix_rot: 三维空间中标准的旋转矩阵

        Returns:
            旋转后的晶格矢量构成的矩阵
        """
        lattice = self.lattice.copy()
        matrix_rot = matrix_rot.reshape(3, 3)

        lattice_rotated = lattice @ matrix_rot.T

        return lattice_rotated
    
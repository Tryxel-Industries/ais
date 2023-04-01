import numpy as np
import pandas as pd


class CellDim:
    def __init__(self, type: str, offset: float, multi: float):
        self.multi = multi
        self.offset = offset
        self.type = type

    def get_value(self, dim_v):
        match self.type:
            case "Circle":
                return ((dim_v-self.offset)*self.multi)**2
            case "Open":
                return (dim_v-self.offset)*self.multi
            case "Disabled":
                return 0
            case other:
                raise Exception()



class BCell:

    def __init__(self,class_label: int, radius: float, dims: [CellDim]):
        self.dims = dims
        self.radius = radius
        self.class_label = class_label

    def get_value(self, dim_vals: [float]):
        if len(dim_vals) != len(self.dims):
            raise Exception()

        roll = -self.radius
        for n, dim_val in enumerate(dim_vals):
            cell_dim = self.dims[n]
            roll += cell_dim.get_value(dim_val)
        return roll

    def show(self):
        print(f"Ag has {len(self.dims)} dims")
        for n,dim in enumerate(self.dims):
            dim:CellDim = dim
            print("dim {:>2} type: {:>10}".format(n, dim.type))

def build_from_csv_df(df: pd.DataFrame):
    b_cells = []
    for idx, row in df.iterrows():
        class_label = row[0]
        radius = row[1]

        remaining = row[2:]
        dims = int(len(remaining)/3)

        cell_dims = []
        for n in range(dims):
            base_idx = 2+n*3
            type = row[base_idx]
            offset = row[base_idx+1]
            multi = row[base_idx+2]
            cell_dims.append(CellDim(type,offset,multi))

        b_cells.append(BCell(class_label, radius, cell_dims))

    return b_cells

x = np.linspace(-5,5,100)


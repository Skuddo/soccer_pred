import pandas as pd

class DataHandler():
    available_modes = ["euro", "laliga", "worldcup"]
    mode: str = None
    data: str = None
    def __init__(self, mode):
        if mode not in self.available_modes:
            print(f"Dataset: {mode} is not an acceptable input")
            print("Options: ", end='')
            for i in self.available_modes:
                print(i, end=', ')
            print()
            return 
        
        self.mode = mode
        self.loadData()

    def loadData(self, path):
        self.data = pd.read_csv(path)

    
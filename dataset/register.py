import numpy as np
import csv






# In all this code : e_* = expressive contour, u_* = unexpressive contour

class Register:
    
    def __init__(self, register_name, folder):
        self.folder = folder + "/"
        self.number_of_file = 0
        self.register_name = register_name
        # create register :
        with open(register_name, 'w') as csvfile:
            fieldnames = ["file_name", "instrument"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        
    

    def add_to_db(self, filename, instrument, u_frequency, u_loudness, e_frequency, e_loudness, verbose = False):
        """ Inputs : unexpressive frequency and loudness, expressive frequency and loudness contours
            Output : None
            Create new CSV file for contours"""

        if not (u_frequency.shape == u_loudness.shape == e_frequency.shape == e_loudness.shape):
            if verbose:
                print("Shapes do not match : \n Unexpressive Contours :\n f0 {} : Loudness {} \n Expressive Contours : \n f0 {} : Loudness {}".format(u_frequency.shape, u_loudness.shape, e_frequency.shape, e_loudness.shape))
                print("Process Stoped")
            return None

        with open(self.folder + filename, 'w') as csvfile:
            fieldnames = ["u_f0", "u_loudness", "e_f0", "e_loudness"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for t in range(e_frequency.shape[0]):
               writer.writerow({"u_f0": str(u_frequency[t]), "u_loudness": str(u_loudness[t]), "e_f0" : str(e_frequency[t]), "e_loudness" : str(e_loudness[t])})

        self.add_to_register(filename, instrument, verbose)


    def add_to_register(self, filename, instrument, verbose = False):
        """ Input : filename
            Output : None
            Add new filename to the register"""

        with open(self.register_name, 'w') as csvfile:
            fieldnames = ["file_name", "instrument"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({"file_name": filename, "instrument": instrument})
        
        self.number_of_file +=1

        if verbose: 
            print("File : {} of {} has been added to the register".format(filename, instrument))
            print("There is now {} files in register".format(self.number_of_file))
        
        
            

        



if __name__ == '__main__':
    r = Register("hihi", "contours-files")
    r.add_to_db("hoho.csv", "violin", np.ones(11),np.ones(11), np.ones(11), np.ones(11), verbose=True)
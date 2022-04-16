import pickle
filename_tda_model = '../TDA_models/IV-PK-Params-in-Humans.pckl'
filehdl_tda_model = open( filename_tda_model , 'rb' )
tda_model = pickle.load(filehdl_tda_model)
filehdl_tda_model.close()

import numpy as np

np.set_printoptions(threshold=np.nan)

inputData_path = '/home/soat/Mystere/data/2018_04_28_full_train-000000-input.npy'
outputData_path1 = '/home/soat/Mystere/data/2018_04_28_full_train-000000-output1.npy'
outputData_path2 = '/home/soat/Mystere/data/2018_04_28_full_train-000000-output2.npy'

Input = np.load(inputData_path, mmap_mode='r+')  # Binary (15444000, 1020)
Output1 = np.load(outputData_path1, mmap_mode='r+')  # Binary (15444000, 2)
Output2 = np.load(outputData_path2, mmap_mode='r+')  # floats (15444000, 30)

for i in range(0, 29):
    v1 = np.array(Output1[1:15000000, 0]).reshape(15000000 - 1, 1).transpose()
    v2 = np.array(Output2[1:15000000, i]).reshape(15000000 - 1, 1).transpose()
    print(np.corrcoef(v1, v2)[0, 1])

    # 9.099654032903703e-05
    # -0.00014141664078351908
    # -0.00042846120215925737
    # -0.00025450262051641205
    # -0.00048402056722246815
    # -0.0007251078591335167
    # -0.0009227516024123605
    # -0.0009344666208173729
    # -0.0007540191587511176
    # -0.0005571168616011339
    # -0.0019706289886565124
    # -0.0013273911935460393
    # -0.0016944500150176947
    # -0.0018363189079652032
    # -0.0017298559266166124
    # -0.00020870271925708305
    # 6.106844596942025e-06
    # 0.00031211711597974436
    # 0.0007009347664339098
    # 0.0005952667071704546
    # 0.001884982044603609
    # 0.0017986331228089592
    # 0.0015502306712475314
    # 0.0015819431520004872
    # 0.002052518621940765
    # 0.0005497284892443
    # 0.0010761470742965155
    # 0.0011635775999824358
    # 0.0008423759713755465

# Take each column of output 2

# Correlate with Output 1 or Just check if the high probabilities are indicators of 1 or 0

# Choose the best column

# print(Input.shape)
#
# print(Input[1, 1:20])
#
# print(Output2[1, 1:30])
#
# print(Input.shape == Output1.shape)
#
# print(Output1.shape)
#
# print(Output2.shape)

# test = np.sum(Output2, axis=1)
#
# print(test)
# /home/cherfaoui_ghiles/mistery/datas/

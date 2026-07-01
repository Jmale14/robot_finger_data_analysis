hp_dict = { "texture_texture":
                {"HP_H_UNITS": 64,
                "HP_FILTERS": 100,
                "HP_KERNEL": 3,
                "HP_POOL": 5,
                "HP_EPOCHS": 150,
                "HP_BATCH": 64,
                "HP_LR": 0.001,
                "HP_L2_LAMBDA": 0.001,
                "HP_LSTM_UNITS": 64 },

            "softness_softness":
                {"HP_H_UNITS": 64,
                "HP_FILTERS": 100,
                "HP_KERNEL": 3,
                "HP_POOL": 5,
                "HP_EPOCHS": 20,
                "HP_BATCH": 64,
                "HP_LR": 0.001,
                "HP_L2_LAMBDA": 0.001,
                "HP_LSTM_UNITS": 64 },

            "text&soft_texture":
                {"HP_H_UNITS": 64,
                "HP_FILTERS": 100,
                "HP_KERNEL": 3,
                "HP_POOL": 5,
                "HP_EPOCHS": 65,
                "HP_BATCH": 64,
                "HP_LR": 0.001,
                "HP_L2_LAMBDA": 0.001,
                "HP_LSTM_UNITS": 64 },

            "text&soft_softness":
                {"HP_H_UNITS": 64,
                "HP_FILTERS": 100,
                "HP_KERNEL": 3,
                "HP_POOL": 5,
                "HP_EPOCHS": 70,
                "HP_BATCH": 64,
                "HP_LR": 0.001,
                "HP_L2_LAMBDA": 0.001,
                "HP_LSTM_UNITS": 64 }
}

abl_epochs = { "texture_texture":
              {"accel": 130,
                "gyro": 170,
               "press": 110},
              
               "softness_softness":
               {"accel": 100,
                 "gyro": 170,
                "press": 50},
               
               "text&soft_texture":
               {"accel": 160,
                 "gyro": 130,
                "press": 150},

                "text&soft_softness":
                {"accel": 170,
                  "gyro": 150,
                 "press": 140}
}
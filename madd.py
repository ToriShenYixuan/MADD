import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def normalized_density_vector(pred_proba_array, nb_decimals, nb_components):
    #print("normalizing density ventor:")
    PP_rounded = np.around(pred_proba_array, decimals=nb_decimals)

    density_vector = np.zeros(nb_components)  # empty
    proba_values = np.linspace(0, 1, nb_components)  # 101 increasing components

    for i in range(len(proba_values)):
        compar = proba_values[i]
        count = 0
        for x in PP_rounded:
            if x == compar:
                count = count + 1
        density_vector[i] = count

    normalized_density_vec = density_vector / np.sum(density_vector)
    #print(np.sum(normalized_density_vec))
    return normalized_density_vec


def MADD_val(norm_densvect_1, norm_densvect_0):
    return np.absolute(norm_densvect_1-norm_densvect_0).sum()

def resample(df_new, demographic):
    df= df_new.copy()
    demographic_unique = df_new[demographic].unique() 
    np.random.shuffle(demographic_unique) 
    # Create a dictionary to map the shuffled labels to the original labels 
    shuffle_mapping = {original_label: shuffled_label for original_label, shuffled_label in zip(df[demographic].unique(), demographic_unique)} 
    df[demographic] = df[demographic].map(shuffle_mapping)
    return df

def get_p(df_new,demographic, bin1, bin2, predicted, nb_decimals, bootstrap, true_madd):
    madd=[]
    for i in range(bootstrap):
        if i%10==0:
            print("bootstrap:",i+1)
        df=resample(df_new, demographic)
        madd_val=MADD(df,demographic, bin1, bin2, predicted, nb_decimals, bootstrap=False, getGraph=False)
        madd.append(madd_val)
    p=len([x for x in madd if x > true_madd])/len(madd)
    return p

def MADD(df, demographic, bin1, bin2, predicted, nb_decimals, bootstrap=False, getGraph=True):
    
    nb_components = 10**nb_decimals+1
    arr1=df.loc[df[demographic]==bin1,predicted]
    if bin2=="all":
        arr2=df[predicted]
    elif bin2=="other":
        arr2=df.loc[df[demographic]!=bin1,predicted]
    else:
        arr2=df.loc[df[demographic]==bin2,predicted]

    norm_densvect_1=normalized_density_vector(arr1, nb_decimals, nb_components)
    norm_densvect_2=normalized_density_vector(arr2, nb_decimals, nb_components)
    madd_val=round(MADD_val(norm_densvect_1, norm_densvect_2), nb_decimals) # the madd value

    if bootstrap:
        df_new=df.copy()
        p=get_p(df_new,demographic, bin1, bin2, predicted, nb_decimals, bootstrap, madd_val)

    if getGraph:
        #plot of density curves
        fig, axs = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(f"Madd output for {df.name} on {demographic}")
        x=np.linspace(0,1,nb_components)
        axs[0].plot(x,norm_densvect_1, label=f"{demographic}={bin1}")
        axs[0].plot(x,norm_densvect_2, label=f"{demographic}={bin2}")
        axs[0].set_xlabel(f"probability sampling bins, e={10**(-nb_decimals)}")
        axs[0].legend()
        title=f"Normalized Density Curves, MADD={madd_val}"
        image_name=f"{demographic}_{bin1}_{bin2}_madd.png"
        if bootstrap: 
            title=f"Normalized Density Curves, MADD={madd_val}, p={p}"
            image_name=f"/{demographic}_{bin1}_{bin2}_{bootstrap}_boot_madd.png"
        axs[0].set_title(title)

        # kernel smoothed visualization output
        mean1=arr1.mean()
        mean2=arr2.mean()

        # plot the 2 DDPs on the same graph
        axs[1] = sns.kdeplot(data=arr1,  label=f"{demographic}={bin1}", clip=(0.0, 1.0))
        axs[1].set_xlabel("Predicted probabilities  [0 ; 1]")

        axs[1] = sns.kdeplot(data=arr2, label=f"{demographic}={bin2}", clip=(0.0, 1.0))
        axs[1].set_ylabel("Density")

        axs[1].vlines(x=mean1, ymin=0, ymax=2.2, color="#1f77b4",linestyle='--')
        axs[1].vlines(x=mean2, ymin=0, ymax=2.2, color="#ff7f0e",linestyle='--')

        axs[1].legend()
        axs[1].set_title("Kernel Smoothed Density Visualizations")
        l1 = axs[1].lines[0]
        l2 = axs[1].lines[1]
        x1 = l1.get_xydata()[:, 0]
        y1 = l1.get_xydata()[:, 1]
        x2 = l2.get_xydata()[:, 0]
        y2 = l2.get_xydata()[:, 1]
        xfill = np.sort(np.concatenate([x1, x2]))
        y1fill = np.interp(xfill, x1, y1)
        y2fill = np.interp(xfill, x2, y2)
        axs[1].fill_between(xfill, y1fill, y2fill, color="#bcbd22")
        path=os.path.join("/Users/shent/Desktop/summer23/fairness/abroca_boot/output/"+df.name)
        image_path=os.path.join(path+image_name)
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)
        print(f"image saved to {image_path}")
        fig.savefig(image_path)
        plt.close()

    return madd_val
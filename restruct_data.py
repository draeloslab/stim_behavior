import os
import cv2
import h5py
import skimage.measure
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from proSVD import proSVD
from scipy.ndimage import gaussian_filter1d


## Translate metadata to csv (do once)
if False: 
    df = pd.read_excel('Movement_class_and_timestamps.xlsx') 
    df2 = df.copy()
    print(df2.keys()[:3])
    df = df.drop(index=0)
    df = df.rename(columns=df2.iloc[0])
    df.to_csv('metadata.csv', index=False)  

## Load csv data
df = pd.read_csv("metadata.csv")

files = [9, 11, 12, 13, 15, 37, 38, 47, 48, 49, 56, 57, 58, 74, 79, 80, 81, 82] #cut 79? no, cut 46
print(len(files))

## grouped figure
fig, ax = plt.subplots(1, 4)
rind = 0
colind = 0

# ## Sumary figure
# fig, ax = plt.subplots(3, 6)
# rind = 0
# colind = 0

# file_num = 79
for i,file_num in enumerate(files):

    if i in [0,1,2,3]:
        colind = 0
    elif i in [7,8,9]:
        colind = 1
    elif i in [10,11,12,13]:
        colind = 2
    elif i in [14,15,16,17]:
        colind = 3        

    if i in [0,1,2,3,7,8,9,10,11,12,13,14,15,16,17]:

        # s = np.load('/media/hawkwings/HDD/octopus_reduced/crop/'+str(df['File Name'][file_num-1])+'_reduced.npy', allow_pickle=True)
        s = np.load('/media/hawkwings/HDD/octopus_reduced/crop/'+str(df['File Name'][file_num-1])+'_data.npy', allow_pickle=True)
        endt = int(df['End Time (s)'][file_num-1] * 30)
        startt = int(df['Start Time (s)'][file_num-1] * 30)

        # x = np.arange(startt-45, endt+90, 1)
        length = (endt+90)-(startt-45)
        # dQ = np.diff(s, axis=2)
        # dQn = np.linalg.norm(dQ, axis=0).T

        # axs.plot(x, np.abs(full_data[1, startt-90 : endt+90]-full_data[1,0]))
        dQn = s.T

        rat = (dQn[startt-45 : endt+90, 1] - dQn[startt-45, 1]) / np.max(dQn[startt-45 : endt+90, 1])
        print(np.max(rat))
        if np.max(rat) > 1:
            breakpoint()

        ax[colind].plot(rat, color='k', alpha=0.5) #, axis=0))
        ax[colind].axvline(x=45, color='g')
        ax[colind].axvline(x=length-90, color='r')
    # fname = str(df['File Name'][file_num-1])
    # ax[rind,colind].title.set_text(fname[:20]+'..'+fname[-15:-3])


    # if colind > 4: 
    #     colind = 0
    #     rind += 1
    # else: colind += 1

plt.tight_layout()
plt.show()

# for file_num in [0]:

    # print('file num is ', file_num)
    # ## For each mp4 input file, read and convert to h5 dataset (do once)
    # if False:
    #     for i in [file_num-1]: #range(df.shape[0]):
    #         # if not os.path.exists('./movies/'+str(df['File Name'][i])+'.h5'):
    #         filename = str('/media/hawkwings/HDD/octopus_movies/crop/'+df['File Name'][i]+'.mp4')
    #         print("file exists?", os.path.exists(filename), 'filename: ', filename)
    #         if not os.path.exists(filename):
    #             # try the -1 format
    #             filename = str('./movies/'+df['File Name'][i]+'-1.mp4')
    #             print("now does the file exist?", os.path.exists(filename), 'filename: ', filename)
    #         cap = cv2.VideoCapture(filename)
    #         length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #         print("movie length", length)
    #         mv = []
    #         if (cap.isOpened()== False): 
    #             print("Error opening video stream or file")
    #             breakpoint()
    #         init = True
    #         while(cap.isOpened()):
    #             ret, frame = cap.read()
    #             # if init: 
    #             #     np.save('/media/hawkwings/HDD/octopus_movies/crop/'+str(df['File Name'][i])+'_image.npy', frame)
    #             #     init = False
    #             # print(frame, ret)
    #             if ret:
    #                 cv2.imshow("frame", frame)
    #                 cv2.waitKey(1)
    #                 # mv.append(np.dot(frame, [0.2989, 0.5870, 0.1140]))
    #             else:
    #                 break

    #         cap.release()
    #         cv2.destroyAllWindows()
    #         print('---- this was number ', i+1)
    #         breakpoint()

    #         # # save as h5 dataset
    #         # mv = np.array(mv)
    #         # # mv = np.dot(mv[...,:3], [0.2989, 0.5870, 0.1140]) #convert to grayscale
    #         # print('movie shape ', mv.shape)
    #         # f = h5py.File('/media/hawkwings/HDD/octopus_movies/crop/'+str(df['File Name'][i])+'.h5', 'w', libver='earliest')
    #         # f.create_dataset("default", data=mv)
    #         # f.close()


    ## Run proSVD
    if True:
        ### ssSVD parameters
        l1 = 90        # num cols (aka time points) to init with
        l = 1           # num cols processed per iter
        decay = 1       # 'forgetting' to track nonstationarity. 1 = no forgetting
        k = 2#l1          # number of singular basis vectors to keep 

        ## testing
        # file_loc = '/home/hawkwings/vorpal_files/StatNeuro/behave/downsampled_videos/frame1.h5'
        # with h5py.File(file_loc, 'r') as f:
        #     d = np.array(f['cam1'])
        #     breakpoint()
        ##

        

        for i in [file_num-1]: #range(df.shape[0]):
            print('Looking at file: ', str(df['File Name'][i]))
            f = h5py.File('/media/hawkwings/HDD/octopus_movies/crop/'+str(df['File Name'][i])+'.h5', 'r')
            d = np.array(f['default'])
            print('movie shape ', d.shape)
            # if d.shape[1] == 480 and d.shape[2] == 640:
            # 480*640 = 307,200 pixels
            data = np.reshape(d, (d.shape[0], d.shape[1]*d.shape[2])).T
            if d.shape[1] == 480 and d.shape[2] == 640:
                factor = int(4)
            elif d.shape[1] >= 1080 and d.shape[2] >= 1920:
                factor = int(8)
            else: factor = int(1)
            del d
            print('factor of ', factor)

            smt = np.zeros(data.shape)
            for filti in range(data.shape[0]):
                smt[filti, :] = gaussian_filter1d(data[filti, :].astype('float'), sigma=2)
            data = smt

            init_data = data[:,:l1]
            
            indata = np.empty((int(init_data.shape[0]/factor), init_data.shape[1]))
            for j in range(l1):
                ## Note: downsampling here
                indata[:,j] = skimage.measure.block_reduce(init_data[:,j], factor, np.mean)
            init_data = indata
            print('init_data shape ', init_data.shape)

            full_size = data.shape[1]
            pro = proSVD(k, w_len=l,history=full_size, decay_alpha=decay, trueSVD=True)
            pro.initialize(init_data)

            print('Finished initialization for proSVD')
            # breakpoint()
            full_data = np.full((k, full_size), np.nan)


            ind = 0 
            # maxi = data.shape[1]-1
            for ii in np.arange(data.shape[1]):
                # for j in np.arange(10):
                di = skimage.measure.block_reduce(data[:,ii], factor, np.mean)[:,None]
                pro.updateSVD(di)

                full_data[:,ind:ind+1] = pro.Qs[:,:,ii].T.dot(di)
                # print(pro.Q.T.shape, di.shape, pro.Q.T.dot(di).shape)
                # print(pro.Qs.shape)
                # if np.any(np.isnan(pro.Q.T.dot(di))):
                #     print('we has a problem.....')
                #     breakpoint()

                # X_rp = reduce_sparseRP(data[:,i:i+1].T, transformer=transformer).T
                # breakpoint()

                ind += 1

            ## save output Qs
            np.save('/media/hawkwings/HDD/octopus_reduced/crop/'+str(df['File Name'][i])+'_reduced.npy', pro.Qs)
            np.save('/media/hawkwings/HDD/octopus_reduced/crop/'+str(df['File Name'][i])+'_data.npy', full_data)

            del pro.Qs
            del full_data
            del data
            # plotting
            # plt.figure()
            # x = np.arange(pro.Qs.shape[2])/30
            # x = np.flip(x, axis=0)
            # endt = df['End Time (s)'][i] 
            # startt = df['Start Time (s)'][i] 
            # qvn = []
            # for i in np.arange(pro.Qs.shape[1]):
            #     qv = pro.Qs[:,i,:]
            #     qvn.append(np.linalg.norm(np.diff(qv,axis=1),axis=0))
            # plt.plot(x[:-1], np.linalg.norm(qvn, axis=0))
            # plt.axvline(x=x.max()-startt, color='g')
            # plt.axvline(x=x.max()-endt, color='r')
            # # plt.gca().invert_xaxis()
            # plt.show()

            # # dQ = np.diff(pro.Qs[:,:10,:], axis=2)
            # # dQn = np.linalg.norm(dQ, axis=0)
            # # plt.plot(x[:-1], dQn.T)
            # # plt.axvline(x=startt, color='g')
            # # plt.axvline(x=endt, color='r')
            # # plt.gca().invert_xaxis()
            # # plt.show()

            # # Qn = np.linalg.norm(pro.Qs[:,:10,:], axis=0)
            # # plt.figure()
            # plt.plot(x, Qn.T)
            # plt.axvline(x=x.max()-startt, color='g')
            # plt.axvline(x=x.max()-endt, color='r')
            # # plt.gca().invert_xaxis()
            # plt.show()



    # ## Examine proSVD basis vectors during/after time of stimulation
    import matplotlib
    matplotlib.use('TkAgg')

    for i in [file_num-1]: #range(df.shape[0]):
        print('Looking at file: ', str(df['File Name'][i]))

        make_movie = True

        try:

            s = np.load('/media/hawkwings/HDD/octopus_reduced/crop/'+str(df['File Name'][i])+'_reduced.npy', allow_pickle=True)
            print('proSVD basis shape: ', s.shape)
            endt = int(df['End Time (s)'][i] * 30) #convert time in seconds to number of frames; acq at 30 frames/sec
            startt = int(df['Start Time (s)'][i] * 30)

            #######
            # endt -= 2*30
            # startt -= 2*30
            #######


            x = np.arange(startt-90, endt+90, 1) #s.shape[2])
            # x = np.flip(x, axis=0)
            Sn = np.linalg.norm(s, axis=0)
            # Sn = Sn - np.max(Sn, axis=1)[:, None]

            # Smax = np.max(np.abs(Sn - np.max(Sn, axis=1)[:, None]), axis=1)
            plt_inds = [0,1] #(-Smax).argsort()[:2]

            fig = plt.figure(figsize=plt.figaspect(0.5))
            axs = fig.add_subplot(1,2,1)
            axs.plot(x, Sn.T[startt-90 : endt+90])
            axs.axvline(x=startt, color='g')
            axs.axvline(x=endt, color='r')
            # plt.show()

            dQ = np.diff(s, axis=2)
            dQn = np.linalg.norm(dQ, axis=0)

            axs = fig.add_subplot(1,2,2)
            axs.plot(x, dQn.T[startt-90 : endt+90])
            axs.axvline(x=startt, color='g')
            axs.axvline(x=endt, color='r')
            
            plt.tight_layout()
            fig.savefig('/media/hawkwings/HDD/octo_results/'+str(df['File Name'][i])+'_basis_stim.svg', bbox_inches='tight')
            # plt.show()

            if make_movie:
                # fig, axs = plt.subplots(ncols=2, figsize=(6, 3), dpi=100)
                fig = plt.figure()
                axs = plt.gca()
                # parameters for animation
                sweep_duration = 15
                hold_duration = 10
                total_duration = sweep_duration + hold_duration
                fps = 15

                # setup animation writer
                import matplotlib.animation
                writer_class = matplotlib.animation.writers['ffmpeg']
                writer = writer_class(fps=fps, bitrate=1000)
                writer.setup(fig, '/media/hawkwings/HDD/octo_results/'+str(df['File Name'][i])+'_res_stim.mp4')

            # f = h5py.File('/media/hawkwings/HDD/octopus_movies/crop/'+str(df['File Name'][i])+'.h5', 'r')
            # d = np.array(f['default'])
            # data = np.reshape(d, (d.shape[0], d.shape[1]*d.shape[2])).T
            # if d.shape[1] == 480 and d.shape[2] == 640:
            #         factor = int(4)
            # elif d.shape[1] >= 1080 and d.shape[2] >= 1920:
            #     factor = int(8)
            # else: factor = int(1)
            # del d 
            
            # l1 = 10  
            # full_data = np.full((l1, data.shape[1]), np.nan)

            full_data = np.load('/media/hawkwings/HDD/octopus_reduced/crop/'+str(df['File Name'][i])+'_data.npy', allow_pickle=True)

            # ind = 0
            if make_movie:
                for ii in np.arange(startt-90, endt+90, 1): #full_data.shape[1]):
                    # di = skimage.measure.block_reduce(data[:,ii], factor, np.mean)[:,None]
                    # full_data[:,ind:ind+1] = s[:,:,ii].T.dot(di)
                    # ind += 1

                    axs.plot(full_data[plt_inds[0],startt-90:ii], full_data[plt_inds[1],startt-90:ii], color='gray', alpha=0.8)
                    if ii >= endt: axs.plot(full_data[plt_inds[0],endt], full_data[plt_inds[1],endt], marker='o', markersize=15, color='r')
                    if ii >= startt: axs.plot(full_data[plt_inds[0],startt], full_data[plt_inds[1],startt], marker='o', markersize=15, color='g')

                    plt.draw()
                    writer.grab_frame()
                    axs.cla()


                writer.finish()

            # np.save('/media/hawkwings/HDD/octopus_reduced/'+str(df['File Name'][i])+'_data.npy', full_data)

            # plt.show()

            fig = plt.figure(figsize=plt.figaspect(0.5))
            axs = fig.add_subplot(1,2,1)
            axs.plot(x, np.abs(full_data[0, startt-90 : endt+90]-full_data[0,0]))
            axs.plot(x, np.abs(full_data[1, startt-90 : endt+90]-full_data[1,0]))
            axs.axvline(x=startt, color='g')
            axs.axvline(x=endt, color='r')
            # plt.show()

            axs = fig.add_subplot(1,2,2)
            axs.plot(x[:-1], np.diff(np.abs(full_data[0, startt-90 : endt+90]-full_data[0,0])))
            axs.plot(x[:-1], np.diff(np.abs(full_data[1, startt-90 : endt+90]-full_data[1,0])))
            axs.axvline(x=startt, color='g')
            axs.axvline(x=endt, color='r')

            plt.tight_layout()
            fig.savefig('/media/hawkwings/HDD/octo_results/'+str(df['File Name'][i])+'_data_stim.svg', bbox_inches='tight')
            # plt.show()

            # breakpoint()

        except:
            import traceback; print(traceback.format_exc())
            pass

        


## Restructure data in better metadata format...

# pull out electrical stimulation ones
# elec = df.loc[~df['Amplitude'].isnull()]
# screen bad videos
# elec = elec.loc[~df['Notes'].isnull()]



breakpoint()
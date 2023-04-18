% A demo script that demonstrates how to process a single video file using
% OpenFace and extract and visualize all of the features

clear

openface_path = "C:\\Users\\ASABUNCUOGLU13\\Documents\\data\\code\\OpenFace_2.2.0_win_x64\\"
executable = openface_path + "FeatureExtraction.exe";
in_dir = "C:\\Users\\ASABUNCUOGLU13\\Documents\\data\\vol04\\face-center\\"
out_dir = "C:\\Users\\ASABUNCUOGLU13\\Documents\\data\\vol04\\face-features\\"


S = dir(fullfile(in_dir,'*'));
N = setdiff({S([S.isdir]).name},{'.','..'}); % list of subfolders of D.
for ii = 1:numel(N)
    T = dir(fullfile(in_dir,N{ii},'*')); % improve by specifying the file extension.
    C = setdiff({T([T.isdir]).name},{'.','..'});
    for jj = 1:numel(C)
        in = fullfile(in_dir, N{ii}, C{jj}); % improve by specifying the file extension.
        out = fullfile(out_dir, N{ii}, C{jj}); % improve by specifying the file extension.
        
        command = sprintf('%s -fdir "%s" -out_dir "%s" -verbose', executable, in, out);
        if(isunix)
            unix(command);
        else
            dos(command);
        end
    end
end
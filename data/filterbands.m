t = 1:1000;
seq = outputmeans(1,:);

hf = 2.^((1:8)/4)/8;
lf = 2.^((0:7)/4)/8;


for i = 1:8
    [B,A] = butter(8,[lf(i) hf(i)]);
    filtseq(:,i) = filtfilt(B,A,seq);
    filtseq(:,i) = abs(hilbert(filtseq(:,i)));
    
    %subplot(8,1,i)
    %plot(filtseq(:,i))
end

figure
surf(filtseq(100:900,2:8))
shading interp

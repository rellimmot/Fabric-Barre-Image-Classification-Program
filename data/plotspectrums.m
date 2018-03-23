f = 1:1000;
spec = abs(fft(outputmeans(1,:)));
subplot(2,1,1)
semilogx(spec(1:100))

spec = abs(fft(outputmeans(9,:)));
subplot(2,1,2)
semilogx(spec(1:100))
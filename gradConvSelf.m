function grad = gradConvSelf(weights, pad, stride, Yprev, gradNext)
  szW = size(weights);
  szYprev = size(Yprev);
  padYprev = zeros(szYprev + 2*pad);
  padYprev(pad+1:pad+szYprev(1), pad+1:pad+szYprev(2), ...
    pad+1:pad+szYprev(3)) = Yprev; 
  grad = zeros(szYprev);
  
  for i = 1:szYprev(1)
    for j = 1:szYprev(2)
      for k = 1:szYprev(3)

        for filter = 1:szW(4)
          for a = 0:szW(1)-1
            for b = 0:szW(2)-1
              for c = 0:szW(3)-1
                %if out of bounds, continue
                p = i-a; q = j-b; r = k-c;
                if p <= 0 || q <= 0 || r<= 0
                  continue;
                end
                grad(i,j,k) = grad(i,j,k) + ...
                  gradNext(p,q,r) * weights(a+1,b+1,c+1,filter); 
              end
            end
          end
        end
      end
    end
  end
end

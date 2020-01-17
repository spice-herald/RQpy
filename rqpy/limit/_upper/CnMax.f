      Real function CnMax(Mmax,f,mu,HighCut,fmin,mtop,Istat)
C Calculates the maximum interval C_n.
C Mmax    Highest m to be used in maximizing C_m.
C f(m)    Probability for each event of it being in the maximum interval
C           with m events
C mu      The expected number of events in the entire experimental range
C HighCut The cut on mu separating low from high statistics
C fmin    The minimum fraction of cumulative probability allowed for seeking
C           the optimum interval
C mtop    The m for which the maximum interval occurs
C Istat   Return code.  Zero = OK, 1 is highly suspicious, and 5 is bad.
C ForCnMax Entry (see below) for producing the f array.
      Integer N,Mmax,Istat,mtop
      Real f(0:Mmax+1),mu,HighCut,ForCnMax
      Integer m,I,Jstat
      Real C,Cinf,y,FC(0:N+1)
      CnMax=-1.
      Istat=0
      Do m=0,Mmax
       If(f(m).gt.fmin) Then
         If(mu.le.HighCut) Then
            C=ConfLev(m,mu*f(m),mu,Jstat)
            If(Jstat.eq.2) Then
C ConfLev failed.
               Istat=5
               Fup=0.
               Return
            ElseIf(Jstat.eq.1) Then
               Istat=1
            EndIf
         Else
            y=(float(m)-mu*f(m))/sqrt(mu*f(m))
            C=Cinf(y,f(m),Istat)
         EndIf
         If(C.gt.CnMax) Then
            CnMax=C
            mtop=m
         EndIf
         If(f(m).ge.1.) Go to 10
       EndIf
      EndDo
 10   Continue
      Return
      Entry ForCnMax(N,FC,f,Mmax)
C Calculates the f array for use above. It uses the set of probabilities
C that an event will be below the each observed one.  In the absence of
C background (assumed here) this can be done once per data set, and CnMax
C can be invoked for multiple values of mu.
C
C FC       Probability of each event being below the observed one
C N        Number of events
C Mmax, f  As for the CnMax entry above
C ForCnMax 0 is OK, 1 is incorrect FC array and therefore incorrect f.
C
C      Write(6,20) N,Mmax,(FC(I),I=0,Mmax+1)
C 20   Format(2I3,20F5.2)
      Do m=0,Mmax
         If( FC(m).lt. 0. .or. FC(m) .gt. 1. .or.
     1    ( m.lt.N .and. FC(m).gt.FC(m+1) ) ) Then
            ForCnMax=1.
            Write(6,*) "ForCnMax finds the FC array to be bad."
            Return
         EndIf
         f(m)=0.
         Do I=0,N-m
C            Write(6,*) m,I
            f(m)=Max(f(m),FC(I+m+1)-FC(I))
         EndDo
      EndDo
      ForCnMax=0.
      End

      SUBROUTINE  DC3D0(ALPHA,X,Y,Z,DEPTH,DIP,POT1,POT2,POT3,POT4,
     *               UX,UY,UZ,UXX,UYX,UZX,UXY,UYY,UZY,UXZ,UYZ,UZZ,IRET)
      IMPLICIT REAL*8 (A-H,O-Z)
      REAL*4   ALPHA,X,Y,Z,DEPTH,DIP,POT1,POT2,POT3,POT4,
     *         UX,UY,UZ,UXX,UYX,UZX,UXY,UYY,UZY,UXZ,UYZ,UZZ
C
C********************************************************************
C*****                                                          *****
C*****    DISPLACEMENT AND STRAIN AT DEPTH                      *****
C*****    DUE TO BURIED POINT SOURCE IN A SEMIINFINITE MEDIUM   *****
C*****                         CODED BY  Y.OKADA ... SEP.1991   *****
C*****                         REVISED     NOV.1991, MAY.2002   *****
C*****                                                          *****
C********************************************************************
C
C***** INPUT
C*****   ALPHA : MEDIUM CONSTANT  (LAMBDA+MYU)/(LAMBDA+2*MYU)
C*****   X,Y,Z : COORDINATE OF OBSERVING POINT
C*****   DEPTH : SOURCE DEPTH
C*****   DIP   : DIP-ANGLE (DEGREE)
C*****   POT1-POT4 : STRIKE-, DIP-, TENSILE- AND INFLATE-POTENCY
C*****       POTENCY=(  MOMENT OF DOUBLE-COUPLE  )/MYU     FOR POT1,2
C*****       POTENCY=(INTENSITY OF ISOTROPIC PART)/LAMBDA  FOR POT3
C*****       POTENCY=(INTENSITY OF LINEAR DIPOLE )/MYU     FOR POT4
C
C***** OUTPUT
C*****   UX, UY, UZ  : DISPLACEMENT ( UNIT=(UNIT OF POTENCY) /
C*****               :                     (UNIT OF X,Y,Z,DEPTH)**2  )
C*****   UXX,UYX,UZX : X-DERIVATIVE ( UNIT= UNIT OF POTENCY) /
C*****   UXY,UYY,UZY : Y-DERIVATIVE        (UNIT OF X,Y,Z,DEPTH)**3  )
C*****   UXZ,UYZ,UZZ : Z-DERIVATIVE
C*****   IRET        : RETURN CODE
C*****               :   =0....NORMAL
C*****               :   =1....SINGULAR
C*****               :   =2....POSITIVE Z WAS GIVEN
C
      COMMON /C1/DUMMY(8),R
      DIMENSION  U(12),DUA(12),DUB(12),DUC(12)
      DATA  F0/0.D0/
C-----
      IRET=0
      IF(Z.GT.0.) THEN
        IRET=2
        GO TO 99
      ENDIF
C-----
      DO 111 I=1,12
        U(I)=F0
        DUA(I)=F0
        DUB(I)=F0
        DUC(I)=F0
  111 CONTINUE
      AALPHA=ALPHA
      DDIP=DIP
      CALL DCCON0(AALPHA,DDIP)
C======================================
C=====  REAL-SOURCE CONTRIBUTION  =====
C======================================
      XX=X
      YY=Y
      ZZ=Z
      DD=DEPTH+Z
      CALL DCCON1(XX,YY,DD)
      IF(R.EQ.F0) THEN
        IRET=1
        GO TO 99
      ENDIF
C-----
      PP1=POT1
      PP2=POT2
      PP3=POT3
      PP4=POT4
      CALL UA0(XX,YY,DD,PP1,PP2,PP3,PP4,DUA)
C-----
      DO 222 I=1,12
        IF(I.LT.10) U(I)=U(I)-DUA(I)
        IF(I.GE.10) U(I)=U(I)+DUA(I)
  222 CONTINUE
C=======================================
C=====  IMAGE-SOURCE CONTRIBUTION  =====
C=======================================
      DD=DEPTH-Z
      CALL DCCON1(XX,YY,DD)
      CALL UA0(XX,YY,DD,PP1,PP2,PP3,PP4,DUA)
      CALL UB0(XX,YY,DD,ZZ,PP1,PP2,PP3,PP4,DUB)
      CALL UC0(XX,YY,DD,ZZ,PP1,PP2,PP3,PP4,DUC)
C-----
      DO 333 I=1,12
        DU=DUA(I)+DUB(I)+ZZ*DUC(I)
        IF(I.GE.10) DU=DU+DUC(I-9)
        U(I)=U(I)+DU
  333 CONTINUE
C=====
      UX=U(1)
      UY=U(2)
      UZ=U(3)
      UXX=U(4)
      UYX=U(5)
      UZX=U(6)
      UXY=U(7)
      UYY=U(8)
      UZY=U(9)
      UXZ=U(10)
      UYZ=U(11)
      UZZ=U(12)
      RETURN
C=======================================
C=====  IN CASE OF SINGULAR (R=0)  =====
C=======================================
   99 UX=F0
      UY=F0
      UZ=F0
      UXX=F0
      UYX=F0
      UZX=F0
      UXY=F0
      UYY=F0
      UZY=F0
      UXZ=F0
      UYZ=F0
      UZZ=F0
      RETURN
      END
      SUBROUTINE  UA0(X,Y,D,POT1,POT2,POT3,POT4,U)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION U(12),DU(12)
C
C********************************************************************
C*****    DISPLACEMENT AND STRAIN AT DEPTH (PART-A)             *****
C*****    DUE TO BURIED POINT SOURCE IN A SEMIINFINITE MEDIUM   *****
C********************************************************************
C
C***** INPUT
C*****   X,Y,D : STATION COORDINATES IN FAULT SYSTEM
C*****   POT1-POT4 : STRIKE-, DIP-, TENSILE- AND INFLATE-POTENCY
C***** OUTPUT
C*****   U(12) : DISPLACEMENT AND THEIR DERIVATIVES
C
      COMMON /C0/ALP1,ALP2,ALP3,ALP4,ALP5,SD,CD,SDSD,CDCD,SDCD,S2D,C2D
      COMMON /C1/P,Q,S,T,XY,X2,Y2,D2,R,R2,R3,R5,QR,QRX,A3,A5,B3,C3,
     *           UY,VY,WY,UZ,VZ,WZ
      DATA F0,F1,F3/0.D0,1.D0,3.D0/
      DATA PI2/6.283185307179586D0/
C-----
      DO 111  I=1,12
  111 U(I)=F0
C======================================
C=====  STRIKE-SLIP CONTRIBUTION  =====
C======================================
      IF(POT1.NE.F0) THEN
        DU( 1)= ALP1*Q/R3    +ALP2*X2*QR
        DU( 2)= ALP1*X/R3*SD +ALP2*XY*QR
        DU( 3)=-ALP1*X/R3*CD +ALP2*X*D*QR
        DU( 4)= X*QR*(-ALP1 +ALP2*(F1+A5) )
        DU( 5)= ALP1*A3/R3*SD +ALP2*Y*QR*A5
        DU( 6)=-ALP1*A3/R3*CD +ALP2*D*QR*A5
        DU( 7)= ALP1*(SD/R3-Y*QR) +ALP2*F3*X2/R5*UY
        DU( 8)= F3*X/R5*(-ALP1*Y*SD +ALP2*(Y*UY+Q) )
        DU( 9)= F3*X/R5*( ALP1*Y*CD +ALP2*D*UY )
        DU(10)= ALP1*(CD/R3+D*QR) +ALP2*F3*X2/R5*UZ
        DU(11)= F3*X/R5*( ALP1*D*SD +ALP2*Y*UZ )
        DU(12)= F3*X/R5*(-ALP1*D*CD +ALP2*(D*UZ-Q) )
        DO 222 I=1,12
  222   U(I)=U(I)+POT1/PI2*DU(I)
      ENDIF
C===================================
C=====  DIP-SLIP CONTRIBUTION  =====
C===================================
      IF(POT2.NE.F0) THEN
        DU( 1)=            ALP2*X*P*QR
        DU( 2)= ALP1*S/R3 +ALP2*Y*P*QR
        DU( 3)=-ALP1*T/R3 +ALP2*D*P*QR
        DU( 4)=                 ALP2*P*QR*A5
        DU( 5)=-ALP1*F3*X*S/R5 -ALP2*Y*P*QRX
        DU( 6)= ALP1*F3*X*T/R5 -ALP2*D*P*QRX
        DU( 7)=                          ALP2*F3*X/R5*VY
        DU( 8)= ALP1*(S2D/R3-F3*Y*S/R5) +ALP2*(F3*Y/R5*VY+P*QR)
        DU( 9)=-ALP1*(C2D/R3-F3*Y*T/R5) +ALP2*F3*D/R5*VY
        DU(10)=                          ALP2*F3*X/R5*VZ
        DU(11)= ALP1*(C2D/R3+F3*D*S/R5) +ALP2*F3*Y/R5*VZ
        DU(12)= ALP1*(S2D/R3-F3*D*T/R5) +ALP2*(F3*D/R5*VZ-P*QR)
        DO 333 I=1,12
  333   U(I)=U(I)+POT2/PI2*DU(I)
      ENDIF
C========================================
C=====  TENSILE-FAULT CONTRIBUTION  =====
C========================================
      IF(POT3.NE.F0) THEN
        DU( 1)= ALP1*X/R3 -ALP2*X*Q*QR
        DU( 2)= ALP1*T/R3 -ALP2*Y*Q*QR
        DU( 3)= ALP1*S/R3 -ALP2*D*Q*QR
        DU( 4)= ALP1*A3/R3     -ALP2*Q*QR*A5
        DU( 5)=-ALP1*F3*X*T/R5 +ALP2*Y*Q*QRX
        DU( 6)=-ALP1*F3*X*S/R5 +ALP2*D*Q*QRX
        DU( 7)=-ALP1*F3*XY/R5           -ALP2*X*QR*WY
        DU( 8)= ALP1*(C2D/R3-F3*Y*T/R5) -ALP2*(Y*WY+Q)*QR
        DU( 9)= ALP1*(S2D/R3-F3*Y*S/R5) -ALP2*D*QR*WY
        DU(10)= ALP1*F3*X*D/R5          -ALP2*X*QR*WZ
        DU(11)=-ALP1*(S2D/R3-F3*D*T/R5) -ALP2*Y*QR*WZ
        DU(12)= ALP1*(C2D/R3+F3*D*S/R5) -ALP2*(D*WZ-Q)*QR
        DO 444 I=1,12
  444   U(I)=U(I)+POT3/PI2*DU(I)
      ENDIF
C=========================================
C=====  INFLATE SOURCE CONTRIBUTION  =====
C=========================================
      IF(POT4.NE.F0) THEN
        DU( 1)=-ALP1*X/R3
        DU( 2)=-ALP1*Y/R3
        DU( 3)=-ALP1*D/R3
        DU( 4)=-ALP1*A3/R3
        DU( 5)= ALP1*F3*XY/R5
        DU( 6)= ALP1*F3*X*D/R5
        DU( 7)= DU(5)
        DU( 8)=-ALP1*B3/R3
        DU( 9)= ALP1*F3*Y*D/R5
        DU(10)=-DU(6)
        DU(11)=-DU(9)
        DU(12)= ALP1*C3/R3
        DO 555 I=1,12
  555   U(I)=U(I)+POT4/PI2*DU(I)
      ENDIF
      RETURN
      END
      SUBROUTINE  UB0(X,Y,D,Z,POT1,POT2,POT3,POT4,U)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION U(12),DU(12)
C
C********************************************************************
C*****    DISPLACEMENT AND STRAIN AT DEPTH (PART-B)             *****
C*****    DUE TO BURIED POINT SOURCE IN A SEMIINFINITE MEDIUM   *****
C********************************************************************
C
C***** INPUT
C*****   X,Y,D,Z : STATION COORDINATES IN FAULT SYSTEM
C*****   POT1-POT4 : STRIKE-, DIP-, TENSILE- AND INFLATE-POTENCY
C***** OUTPUT
C*****   U(12) : DISPLACEMENT AND THEIR DERIVATIVES
C
      COMMON /C0/ALP1,ALP2,ALP3,ALP4,ALP5,SD,CD,SDSD,CDCD,SDCD,S2D,C2D
      COMMON /C1/P,Q,S,T,XY,X2,Y2,D2,R,R2,R3,R5,QR,QRX,A3,A5,B3,C3,
     *           UY,VY,WY,UZ,VZ,WZ
      DATA F0,F1,F2,F3,F4,F5,F8,F9
     *        /0.D0,1.D0,2.D0,3.D0,4.D0,5.D0,8.D0,9.D0/
      DATA PI2/6.283185307179586D0/
C-----
      C=D+Z
      RD=R+D
      D12=F1/(R*RD*RD)
      D32=D12*(F2*R+D)/R2
      D33=D12*(F3*R+D)/(R2*RD)
      D53=D12*(F8*R2+F9*R*D+F3*D2)/(R2*R2*RD)
      D54=D12*(F5*R2+F4*R*D+D2)/R3*D12
C-----
      FI1= Y*(D12-X2*D33)
      FI2= X*(D12-Y2*D33)
      FI3= X/R3-FI2
      FI4=-XY*D32
      FI5= F1/(R*RD)-X2*D32
      FJ1=-F3*XY*(D33-X2*D54)
      FJ2= F1/R3-F3*D12+F3*X2*Y2*D54
      FJ3= A3/R3-FJ2
      FJ4=-F3*XY/R5-FJ1
      FK1=-Y*(D32-X2*D53)
      FK2=-X*(D32-Y2*D53)
      FK3=-F3*X*D/R5-FK2
C-----
      DO 111  I=1,12
  111 U(I)=F0
C======================================
C=====  STRIKE-SLIP CONTRIBUTION  =====
C======================================
      IF(POT1.NE.F0) THEN
        DU( 1)=-X2*QR  -ALP3*FI1*SD
        DU( 2)=-XY*QR  -ALP3*FI2*SD
        DU( 3)=-C*X*QR -ALP3*FI4*SD
        DU( 4)=-X*QR*(F1+A5) -ALP3*FJ1*SD
        DU( 5)=-Y*QR*A5      -ALP3*FJ2*SD
        DU( 6)=-C*QR*A5      -ALP3*FK1*SD
        DU( 7)=-F3*X2/R5*UY      -ALP3*FJ2*SD
        DU( 8)=-F3*XY/R5*UY-X*QR -ALP3*FJ4*SD
        DU( 9)=-F3*C*X/R5*UY     -ALP3*FK2*SD
        DU(10)=-F3*X2/R5*UZ  +ALP3*FK1*SD
        DU(11)=-F3*XY/R5*UZ  +ALP3*FK2*SD
        DU(12)= F3*X/R5*(-C*UZ +ALP3*Y*SD)
        DO 222 I=1,12
  222   U(I)=U(I)+POT1/PI2*DU(I)
      ENDIF
C===================================
C=====  DIP-SLIP CONTRIBUTION  =====
C===================================
      IF(POT2.NE.F0) THEN
        DU( 1)=-X*P*QR +ALP3*FI3*SDCD
        DU( 2)=-Y*P*QR +ALP3*FI1*SDCD
        DU( 3)=-C*P*QR +ALP3*FI5*SDCD
        DU( 4)=-P*QR*A5 +ALP3*FJ3*SDCD
        DU( 5)= Y*P*QRX +ALP3*FJ1*SDCD
        DU( 6)= C*P*QRX +ALP3*FK3*SDCD
        DU( 7)=-F3*X/R5*VY      +ALP3*FJ1*SDCD
        DU( 8)=-F3*Y/R5*VY-P*QR +ALP3*FJ2*SDCD
        DU( 9)=-F3*C/R5*VY      +ALP3*FK1*SDCD
        DU(10)=-F3*X/R5*VZ -ALP3*FK3*SDCD
        DU(11)=-F3*Y/R5*VZ -ALP3*FK1*SDCD
        DU(12)=-F3*C/R5*VZ +ALP3*A3/R3*SDCD
        DO 333 I=1,12
  333   U(I)=U(I)+POT2/PI2*DU(I)
      ENDIF
C========================================
C=====  TENSILE-FAULT CONTRIBUTION  =====
C========================================
      IF(POT3.NE.F0) THEN
        DU( 1)= X*Q*QR -ALP3*FI3*SDSD
        DU( 2)= Y*Q*QR -ALP3*FI1*SDSD
        DU( 3)= C*Q*QR -ALP3*FI5*SDSD
        DU( 4)= Q*QR*A5 -ALP3*FJ3*SDSD
        DU( 5)=-Y*Q*QRX -ALP3*FJ1*SDSD
        DU( 6)=-C*Q*QRX -ALP3*FK3*SDSD
        DU( 7)= X*QR*WY     -ALP3*FJ1*SDSD
        DU( 8)= QR*(Y*WY+Q) -ALP3*FJ2*SDSD
        DU( 9)= C*QR*WY     -ALP3*FK1*SDSD
        DU(10)= X*QR*WZ +ALP3*FK3*SDSD
        DU(11)= Y*QR*WZ +ALP3*FK1*SDSD
        DU(12)= C*QR*WZ -ALP3*A3/R3*SDSD
        DO 444 I=1,12
  444   U(I)=U(I)+POT3/PI2*DU(I)
      ENDIF
C=========================================
C=====  INFLATE SOURCE CONTRIBUTION  =====
C=========================================
      IF(POT4.NE.F0) THEN
        DU( 1)= ALP3*X/R3
        DU( 2)= ALP3*Y/R3
        DU( 3)= ALP3*D/R3
        DU( 4)= ALP3*A3/R3
        DU( 5)=-ALP3*F3*XY/R5
        DU( 6)=-ALP3*F3*X*D/R5
        DU( 7)= DU(5)
        DU( 8)= ALP3*B3/R3
        DU( 9)=-ALP3*F3*Y*D/R5
        DU(10)=-DU(6)
        DU(11)=-DU(9)
        DU(12)=-ALP3*C3/R3
        DO 555 I=1,12
  555   U(I)=U(I)+POT4/PI2*DU(I)
      ENDIF
      RETURN
      END
      SUBROUTINE  UC0(X,Y,D,Z,POT1,POT2,POT3,POT4,U)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION U(12),DU(12)
C
C********************************************************************
C*****    DISPLACEMENT AND STRAIN AT DEPTH (PART-B)             *****
C*****    DUE TO BURIED POINT SOURCE IN A SEMIINFINITE MEDIUM   *****
C********************************************************************
C
C***** INPUT
C*****   X,Y,D,Z : STATION COORDINATES IN FAULT SYSTEM
C*****   POT1-POT4 : STRIKE-, DIP-, TENSILE- AND INFLATE-POTENCY
C***** OUTPUT
C*****   U(12) : DISPLACEMENT AND THEIR DERIVATIVES
C
      COMMON /C0/ALP1,ALP2,ALP3,ALP4,ALP5,SD,CD,SDSD,CDCD,SDCD,S2D,C2D
      COMMON /C1/P,Q,S,T,XY,X2,Y2,D2,R,R2,R3,R5,QR,QRX,A3,A5,B3,C3
      DATA F0,F1,F2,F3,F5,F7,F10,F15
     *        /0.D0,1.D0,2.D0,3.D0,5.D0,7.D0,10.D0,15.D0/
      DATA PI2/6.283185307179586D0/
C-----
      C=D+Z
      Q2=Q*Q
      R7=R5*R2
      A7=F1-F7*X2/R2
      B5=F1-F5*Y2/R2
      B7=F1-F7*Y2/R2
      C5=F1-F5*D2/R2
      C7=F1-F7*D2/R2
      D7=F2-F7*Q2/R2
      QR5=F5*Q/R2
      QR7=F7*Q/R2
      DR5=F5*D/R2
C-----
      DO 111  I=1,12
  111 U(I)=F0
C======================================
C=====  STRIKE-SLIP CONTRIBUTION  =====
C======================================
      IF(POT1.NE.F0) THEN
        DU( 1)=-ALP4*A3/R3*CD  +ALP5*C*QR*A5
        DU( 2)= F3*X/R5*( ALP4*Y*CD +ALP5*C*(SD-Y*QR5) )
        DU( 3)= F3*X/R5*(-ALP4*Y*SD +ALP5*C*(CD+D*QR5) )
        DU( 4)= ALP4*F3*X/R5*(F2+A5)*CD   -ALP5*C*QRX*(F2+A7)
        DU( 5)= F3/R5*( ALP4*Y*A5*CD +ALP5*C*(A5*SD-Y*QR5*A7) )
        DU( 6)= F3/R5*(-ALP4*Y*A5*SD +ALP5*C*(A5*CD+D*QR5*A7) )
        DU( 7)= DU(5)
        DU( 8)= F3*X/R5*( ALP4*B5*CD -ALP5*F5*C/R2*(F2*Y*SD+Q*B7) )
        DU( 9)= F3*X/R5*(-ALP4*B5*SD +ALP5*F5*C/R2*(D*B7*SD-Y*C7*CD) )
        DU(10)= F3/R5*   (-ALP4*D*A5*CD +ALP5*C*(A5*CD+D*QR5*A7) )
        DU(11)= F15*X/R7*( ALP4*Y*D*CD  +ALP5*C*(D*B7*SD-Y*C7*CD) )
        DU(12)= F15*X/R7*(-ALP4*Y*D*SD  +ALP5*C*(F2*D*CD-Q*C7) )
        DO 222 I=1,12
  222   U(I)=U(I)+POT1/PI2*DU(I)
      ENDIF
C===================================
C=====  DIP-SLIP CONTRIBUTION  =====
C===================================
      IF(POT2.NE.F0) THEN
        DU( 1)= ALP4*F3*X*T/R5          -ALP5*C*P*QRX
        DU( 2)=-ALP4/R3*(C2D-F3*Y*T/R2) +ALP5*F3*C/R5*(S-Y*P*QR5)
        DU( 3)=-ALP4*A3/R3*SDCD         +ALP5*F3*C/R5*(T+D*P*QR5)
        DU( 4)= ALP4*F3*T/R5*A5              -ALP5*F5*C*P*QR/R2*A7
        DU( 5)= F3*X/R5*(ALP4*(C2D-F5*Y*T/R2)-ALP5*F5*C/R2*(S-Y*P*QR7))
        DU( 6)= F3*X/R5*(ALP4*(F2+A5)*SDCD   -ALP5*F5*C/R2*(T+D*P*QR7))
        DU( 7)= DU(5)
        DU( 8)= F3/R5*(ALP4*(F2*Y*C2D+T*B5)
     *                               +ALP5*C*(S2D-F10*Y*S/R2-P*QR5*B7))
        DU( 9)= F3/R5*(ALP4*Y*A5*SDCD-ALP5*C*((F3+A5)*C2D+Y*P*DR5*QR7))
        DU(10)= F3*X/R5*(-ALP4*(S2D-T*DR5) -ALP5*F5*C/R2*(T+D*P*QR7))
        DU(11)= F3/R5*(-ALP4*(D*B5*C2D+Y*C5*S2D)
     *                                -ALP5*C*((F3+A5)*C2D+Y*P*DR5*QR7))
        DU(12)= F3/R5*(-ALP4*D*A5*SDCD-ALP5*C*(S2D-F10*D*T/R2+P*QR5*C7))
        DO 333 I=1,12
  333   U(I)=U(I)+POT2/PI2*DU(I)
      ENDIF
C========================================
C=====  TENSILE-FAULT CONTRIBUTION  =====
C========================================
      IF(POT3.NE.F0) THEN
        DU( 1)= F3*X/R5*(-ALP4*S +ALP5*(C*Q*QR5-Z))
        DU( 2)= ALP4/R3*(S2D-F3*Y*S/R2)+ALP5*F3/R5*(C*(T-Y+Y*Q*QR5)-Y*Z)
        DU( 3)=-ALP4/R3*(F1-A3*SDSD)   -ALP5*F3/R5*(C*(S-D+D*Q*QR5)-D*Z)
        DU( 4)=-ALP4*F3*S/R5*A5 +ALP5*(C*QR*QR5*A7-F3*Z/R5*A5)
        DU( 5)= F3*X/R5*(-ALP4*(S2D-F5*Y*S/R2)
     *                               -ALP5*F5/R2*(C*(T-Y+Y*Q*QR7)-Y*Z))
        DU( 6)= F3*X/R5*( ALP4*(F1-(F2+A5)*SDSD)
     *                               +ALP5*F5/R2*(C*(S-D+D*Q*QR7)-D*Z))
        DU( 7)= DU(5)
        DU( 8)= F3/R5*(-ALP4*(F2*Y*S2D+S*B5)
     *                -ALP5*(C*(F2*SDSD+F10*Y*(T-Y)/R2-Q*QR5*B7)+Z*B5))
        DU( 9)= F3/R5*( ALP4*Y*(F1-A5*SDSD)
     *                +ALP5*(C*(F3+A5)*S2D-Y*DR5*(C*D7+Z)))
        DU(10)= F3*X/R5*(-ALP4*(C2D+S*DR5)
     *               +ALP5*(F5*C/R2*(S-D+D*Q*QR7)-F1-Z*DR5))
        DU(11)= F3/R5*( ALP4*(D*B5*S2D-Y*C5*C2D)
     *               +ALP5*(C*((F3+A5)*S2D-Y*DR5*D7)-Y*(F1+Z*DR5)))
        DU(12)= F3/R5*(-ALP4*D*(F1-A5*SDSD)
     *               -ALP5*(C*(C2D+F10*D*(S-D)/R2-Q*QR5*C7)+Z*(F1+C5)))
        DO 444 I=1,12
  444   U(I)=U(I)+POT3/PI2*DU(I)
      ENDIF
C=========================================
C=====  INFLATE SOURCE CONTRIBUTION  =====
C=========================================
      IF(POT4.NE.F0) THEN
        DU( 1)= ALP4*F3*X*D/R5
        DU( 2)= ALP4*F3*Y*D/R5
        DU( 3)= ALP4*C3/R3
        DU( 4)= ALP4*F3*D/R5*A5
        DU( 5)=-ALP4*F15*XY*D/R7
        DU( 6)=-ALP4*F3*X/R5*C5
        DU( 7)= DU(5)
        DU( 8)= ALP4*F3*D/R5*B5
        DU( 9)=-ALP4*F3*Y/R5*C5
        DU(10)= DU(6)
        DU(11)= DU(9)
        DU(12)= ALP4*F3*D/R5*(F2+C5)
        DO 555 I=1,12
  555   U(I)=U(I)+POT4/PI2*DU(I)
      ENDIF
      RETURN
      END


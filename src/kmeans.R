#!/usr/bin/env Rscript
args<-commandArgs(TRUE)

whichk <- args[1]

library( ANTsRNet )
library( ANTsR )
library( patchMatchR )
library( tensorflow )
library( keras )

mytype = "float32"

orinetInference <- function( template, target, mdlfn, newway=FALSE, verbose=TRUE ) {

  # rotation template
  newspc = antsGetSpacing( template ) * 2.0
  reoTemplateOrinet = resampleImage( template, newspc ) %>%
    iMath("PadImage", 32 )%>%
    iMath("Normalize")

  orinet = createResNetModel3D(
      list(NULL,NULL,NULL,1),
      inputScalarsSize = 0,
      numberOfClassificationLabels = 6,
      layers = 1:4,
      residualBlockSchedule = c(3, 4, 6, 3),
      lowestResolution = 64,
      cardinality = 1,
      squeezeAndExcite = TRUE,
      mode = "regression"
    )
  load_model_weights_hdf5( orinet, mdlfn )

  myaff = randomAffineImage( reoTemplateOrinet, "Rigid", sdAffine = 0 )[[2]]
  idparams = getAntsrTransformParameters( myaff )
  fixparams = getAntsrTransformFixedParameters( myaff )
  templateCoM = getCenterOfMass( reoTemplateOrinet )


  # BEGIN: a critical section of code that should match the training code
  if ( newway ) {
    lowerTrunc=1e-6
    upperTrunc=0.95
    oimg = smoothImage( target, 1.5, sigmaInPhysicalCoordinates = FALSE ) %>%
      iMath("TruncateIntensity",lowerTrunc,upperTrunc) %>% iMath("Normalize")
    # oimg = iMath( target, "TruncateIntensity", 0.01, 0.99 ) %>% iMath("Normalize")
    imglowreg = antsRegistration( reoTemplateOrinet, oimg, 'Translation' )
    initialTx = imglowreg$fwdtransforms
    orinetimg = iMath( imglowreg$warpedmovout, "Normalize" )
  } else {
    oimg = iMath( target, "TruncateIntensity", 0.01, 0.99 ) %>% iMath("Normalize")
    imglowreg = antsRegistration( reoTemplateOrinet, oimg, 'Translation' )
    initialTx = imglowreg$fwdtransforms
    imglow = imglowreg$warpedmovout
    orinetimg = smoothImage( imglow, 1.5, sigmaInPhysicalCoordinates = FALSE ) %>% iMath("Normalize")
  }
  # END: a critical section of code that should match the training code

  imgCoM = getCenterOfMass( iMath(orinetimg, "Normalize") )
  imgarr = array( as.array( iMath(orinetimg, "Normalize") ), dim=c(1,dim(orinetimg),1) )
  with(tf$device("/cpu:0"), {
    predRot <- predict( orinet, tf$cast( imgarr, mytype), batch_size = 1 )
    })

  mm = matrix( predRot[1,], nrow=3, byrow=F)
  mmm = cbind( mm, pracma::cross( mm[,1], mm[,2] ) )
  mm = polarDecomposition( mmm )$Z
  locparams = getAntsrTransformParameters( myaff )
  locparams[1:9] = mm
  locparams[10:length(locparams)] = (imgCoM - templateCoM )
  setAntsrTransformParameters( myaff, locparams )
  setAntsrTransformFixedParameters( myaff, templateCoM )
  # myaff = composeAntsrTransforms( list( myaff, initialTx ) )
  # rotated = applyAntsrTransformToImage( myaff, orinetimg, reoTemplate )
  if ( verbose ) {
    print("classical rigid")
    message("WE ARE RUNNING NEW ORINET" )
    }
  qreg = antsRegistration( reoTemplateOrinet, imglow, "Rigid", initialTransform=myaff )
  qreg = antsRegistration( reoTemplateOrinet, imglow, "Similarity", initialTransform=qreg$fwdtransforms )
  qreg = antsRegistration( reoTemplateOrinet, imglow, "Affine", initialTransform=qreg$fwdtransforms )
  img2LM = qreg$warpedmovout
  mymi = antsImageMutualInformation( reoTemplateOrinet, img2LM)
  if ( verbose )
    print( paste( "MI:", mymi ) )
  return( c( qreg$fwdtransforms, initialTx )  )
  }

#

km <- function( xIn , mask, k = 3, ... ) {
  x = iMath( xIn, "Normalize" ) * 255
  cl <- kmeans( x[ mask == 1 ] ,  k, iter.max = 2000, nstart = 25, ... )
  segs = makeImage( mask, cl$cluster )
  lstats = labelStats( x, segs )[-1,]
  neword = order( lstats$Mean )
  clusterOrd = cl$cluster * 0
  for ( j in 1:k )
    clusterOrd[ cl$cluster == neword[j] ] = j
  segs = makeImage( mask, clusterOrd )
  return( list( segs, labelStats( x, segs )[-1,]$Mean ) )
}


kmSkull <- function( xIn , smoo=0.5 ) {
  x = smoothImage( xIn, smoo, sigmaInPhysicalCoordinates=FALSE ) %>%
    iMath("Normalize")
  mykm = km( x,  mask=thresholdImage(x,0.01,1), k=4 )
  dct = 0
  while ( mykm[[2]][3] < 110  & dct < 3 ) {
    smoo = smoo * 1.1
    print(paste( "KICKIT", smoo, " thresh: ", mykm[[2]][3] ) )
    x = smoothImage( xIn, smoo, sigmaInPhysicalCoordinates=FALSE ) %>%
      iMath("Normalize")
    mykm = km( x,  mask=thresholdImage(x,0.01,1), k=4 )
    dct = dct + 1
  }
  skullseg = thresholdImage( mykm[[1]], 3,4 )
  origThresh = quantile( xIn[ skullseg == 1 ],0.02)
  skullseg2 = thresholdImage( xIn,origThresh, Inf ) %>% labelClusters( 10 ) %>% thresholdImage( 1, Inf)
  print( paste("T1:",mykm[[2]][3],'T2:',origThresh) )
  return( skullseg )
}


reoTemplate = antsImageRead( "templateImage.nii.gz" )
ptTemplate = data.matrix( read.csv( "templatePoints.csv" ) )
orinetmdlfn =  "models/mouse_rotation_3D_GPU_update1locsd5.h5"
fns = Sys.glob("bigdata/*rec.nii.gz")
myids = basename( fns ) %>% tools::file_path_sans_ext(T)
demog = data.frame( ids = myids,
  fnimg=as.character(fns),
  fnseg=gsub( ".nii.gz","-LM.nii.gz", as.character(fns) ) )


# define an image template that lets us penalize rotations away from this reference
reoTemplate = antsImageRead( "templateImage.nii.gz" )
newspc = antsGetSpacing( reoTemplate ) * 1.
reoTemplate = resampleImage( reoTemplate, newspc ) %>%
  iMath("PadImage", 8 )%>%
  iMath("Normalize")

lowerTrunc=1e-6
upperTrunc=0.98
if ( is.na( whichk ) ) whichk = sample( 1:nrow( demog ) )
for ( k in whichk ) {
  myfn = demog$fnimg[k]
  temper = tools::file_path_sans_ext( basename( myfn ), TRUE )
  outprefix = paste0( "preprocessed/", temper, "-pro-" )
  postfix = c( "reorient.nii.gz", "skull.nii.gz", "skulldist.nii.gz", "pts.nii.gz", "pts.csv" )
  outnames = paste0( outprefix, postfix )
  print( paste( k , myfn , outnames[1] ) )
  if ( any( ! file.exists( outnames ) ) ) {
    oimg = antsImageRead( myfn )
    seg = antsImageRead( demog$fnseg[k] )
    pts = getCentroids( seg )[,1:oimg@dimension]
    fittx = fitTransformToPairedPoints( pts, ptTemplate, "Rigid" )
    mapped2gether = applyAntsrTransformToImage( fittx$transform, oimg, reoTemplate  )
    loctxi = invertAntsrTransform( fittx$transform )
    blobs = applyAntsrTransformToPoint( loctxi, pts )
#    transform = orinetInference( reoTemplate, oimg, orinetmdlfn, newway=FALSE, verbose=FALSE )
#    mapped2gether = antsApplyTransforms( reoTemplate, oimg, transform )
#    plot( reoTemplate, mapped2gether, axis=3, nslices=21, ncolumns=7, alpha=0.5, outname=paste0( outprefix, "mapped.png" ) )
    print( paste( 'MI', antsImageMutualInformation( reoTemplate, mapped2gether ) ) )
    demog[k,'MI']=antsImageMutualInformation( reoTemplate, mapped2gether )
    img = denoiseImage( oimg, noiseModel='Gaussian')
    img = iMath( img, "TruncateIntensity", lowerTrunc, upperTrunc ) %>% iMath("Normalize")
    fgbg = kmSkull( img, 0.5 )
    # skull = antsApplyTransforms( reoTemplate, fgbg, transform, interpolator='nearestNeighbor' )
    skull = applyAntsrTransformToImage( fittx$transform, fgbg, reoTemplate , interpolation='nearestNeighbor'  )
    demog[k,'SkullVol']=sum( skull ) * prod( antsGetSpacing( skull ) )
    plot( mapped2gether, skull, nslices=21, ncol=7, axis=1, outname=paste0( outprefix, "skull.png" ) )
    # now do points
#    rimg = antsApplyTransforms( reoTemplate, img, transform )
    rimg = applyAntsrTransformToImage( fittx$transform, oimg, reoTemplate  )
#    blobs = antsApplyTransformsToPoints( 3, pts, rev(transform), c(TRUE,TRUE))
    mpi = makePointsImage( blobs, rimg*0+1, 0.2 )
#    plot(rimg, mpi, nslices=21, ncol=7, axis=1 )
    maskdist = iMath( skull, 'D' )
    antsImageWrite( rimg, outnames[1] )
    antsImageWrite( skull, outnames[2] )
    antsImageWrite( maskdist, outnames[3] )
    antsImageWrite( mpi, outnames[4] )
    # plot(rimg, mpi, nslices=21, ncol=7, axis=1 )
    write.csv( blobs, outnames[5], row.names=FALSE )
    print( demog[k,] )
    write.csv( demog , 'preprocessed.csv', row.names=FALSE )
  }
}

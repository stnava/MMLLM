Sys.setenv("TF_NUM_INTEROP_THREADS"=12)
Sys.setenv("TF_NUM_INTRAOP_THREADS"=12)
Sys.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"=12)
Sys.setenv(CUDA_VISIBLE_DEVICES=2)
mygpu=Sys.getenv("CUDA_VISIBLE_DEVICES")


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


mdlfn = 'models/autopointsupdate_192_weights_3d_GPUsigmoidHR2.h5'
orinetmdlfn =  "models/mouse_rotation_3D_GPU_update1locsd5.h5"
if ( file.exists( trnhfn ) ) {
  print("TRH")
  trnh = read.csv( trnhfn )
  plot( ts(trnh$loss ))
#  layout( matrix(1:2,nrow=1))
  points( trnh$testErr, col='red')
}

library( ANTsRNet )
library( ANTsR )
library( patchMatchR )
library( tensorflow )
library( keras )
library( reticulate )
library( ggplot2 )
library( tfdatasets )
library( glue )
mytype = "float32"


# define an image template that lets us penalize rotations away from this reference
reoTemplate = antsImageRead( "templateImage.nii.gz" ) %>%
  iMath("PadImage", 8 )%>%
  iMath("Normalize")
ptTemplate = data.matrix( read.csv( "templatePoints.csv" ) )
locdim = dim(reoTemplate)


unet = createUnetModel3D(
       list( NULL, NULL, NULL, 1),
       numberOfOutputs = 55,
       numberOfLayers = 5,
       numberOfFiltersAtBaseLayer = 32,
       convolutionKernelSize = 3,
       deconvolutionKernelSize = 2,
       poolSize = 2,
       strides = 2,
       dropoutRate = 0,
       weightDecay = 0,
       additionalOptions = "nnUnetActivationStyle",
       mode = c("regression")
     )
# unet = keras_model( unet$inputs, layer_multiply( list( unet$output, unet$input ) ) )
findpoints = deepLandmarkRegressionWithHeatmaps( unet, activation='sigmoid', theta=NA )


tefns = Sys.glob( 'bigdata/*rec.nii.gz' )
testfn = sample(tefns,1)
print( testfn )

lmfn = gsub( "rec.nii.gz", "rec-LM.nii.gz", testfn )
oimg = antsImageRead( testfn ) %>% iMath("Normalize")

transform = orinetInference( reoTemplate, oimg, orinetmdlfn, newway=FALSE, verbose=FALSE )
img2LM = antsApplyTransforms( reoTemplate, oimg, transform )
# plot( reoTemplate, img2LM, axis=3, nslices=21, ncolumns=7, alpha=0.5 )

oimgpts = iMath(img2LM,"TruncateIntensity",0.01,0.99) %>% histogramMatchImage( reoTemplate )

img2LMcoords = coordinateImages( img2LM * 0 + 1 )
mycc = array( dim = c( 1, dim( img2LM ), 3 ) )
for ( jj in 1:3 ) mycc[1,,,,jj] = as.array( img2LMcoords[[jj]] )
imgarr = array( as.array( oimgpts ), dim=c(1,dim(img2LM),1 ) )
telist = list(  tf$cast( imgarr, mytype), tf$cast( mycc, mytype) )
load_model_weights_hdf5( findpoints,   mdlfn )
with(tf$device("/cpu:0"), {
      pointsoutte <- predict( findpoints, telist, batch_size = 1 )
      })
ptp = as.array(pointsoutte[[2]])[1,,]
ptimg = makePointsImage( ptp, img2LM*0+1, radius=0.2 )  %>% iMath("GD",1)
print(length(sort(unique(ptimg))))
# plot( img2LM, ptimg, nslices = 21, ncolumns = 7, axis=3 )

antsImageWrite( img2LM, '/tmp/temp.nii.gz' )
antsImageWrite( ptimg, '/tmp/tempt.nii.gz' )

if ( file.exists( lmfn ) ) {
  # the ground truth error
  trulmsIn = getCentroids( antsImageRead(lmfn) )[,1:3]
  trulms = data.matrix( antsApplyTransformsToPoints( 3, trulmsIn, rev(transform),c(T,T) ) )
  distancesByPoint = rep( NA, nrow( trulms ) )
  for ( k in 1:nrow( ptpb ) ) {
    distancesByPoint[k] = sqrt( sum( ( as.numeric(trulms[k,]) - as.numeric(ptp[k,])  )^2 ) )
  }
  print( distancesByPoint )
  print( mean( distancesByPoint ) )
  # roughly percent errror
  print( norm( ( trulms- data.matrix(ptp) ))/norm(trulms) )
}

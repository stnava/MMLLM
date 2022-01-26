Sys.setenv("TF_NUM_INTEROP_THREADS"=12)
Sys.setenv("TF_NUM_INTRAOP_THREADS"=12)
Sys.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"=12)
Sys.setenv(CUDA_VISIBLE_DEVICES=2)
mygpu=Sys.getenv("CUDA_VISIBLE_DEVICES")
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


reoTemplate = antsImageRead( "templateImage.nii.gz" )
orinetmdlfn =  "models/mouse_rotation_3D_GPU_update1locsd5.h5"
tefns = Sys.glob( 'bigdata/*rec.nii.gz' )
testfn = sample(tefns,1)
# testfn="bigdata/skull_157_11__rec.nii.gz"
if ( ! exists('evaldf') ) {
  evaldf = data.frame( IDs=basename(tefns), MI=NA  )
  } else evaldf = read.csv( "/tmp/evaldf.csv" )
for ( k in 1:length( tefns ) ) {
  oimg = antsImageRead( tefns[k] )
  transform = orinetInference( reoTemplate, oimg, orinetmdlfn, newway=FALSE, verbose=FALSE )
  mapped2gether = antsApplyTransforms( reoTemplate, oimg, transform )
  plot( reoTemplate, mapped2gether, axis=3, nslices=21, ncolumns=7, alpha=0.5 )
  evaldf[k,'MI']=antsImageMutualInformation( reoTemplate, mapped2gether )
  print( evaldf[k,] )
  write.csv( evaldf, "/tmp/evaldf.csv", row.names=FALSE )
}

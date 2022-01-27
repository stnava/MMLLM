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

orinetInference <- function( template, target, mdlfn, doreo=NULL, newway=FALSE, verbose=TRUE ) {

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
  } else {
    oimg = iMath( target, "TruncateIntensity", 0.01, 0.99 ) %>% iMath("Normalize")
    if ( is.null( doreo ) ) {
      imglowreg = antsRegistration( reoTemplateOrinet, oimg, 'Translation' )
    } else {
      reo = reorientImage( oimg , axis1 = doreo )
      imglowreg = antsRegistration( reoTemplateOrinet, oimg, 'Translation', initialTransform=reo$txfn )
    }
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
  qreg = antsRegistration( reoTemplateOrinet, imglow, "Rigid", initialTransform=myaff )
  qreg = antsRegistration( reoTemplateOrinet, imglow, "Similarity", initialTransform=qreg$fwdtransforms )
  qreg = antsRegistration( reoTemplateOrinet, imglow, "Affine", initialTransform=qreg$fwdtransforms )
  img2LM = qreg$warpedmovout
  mymi = mean( abs( iMath(reoTemplateOrinet,"Normalize") - iMath(img2LM,"Normalize") ) )
  if ( verbose )
    print( paste( "MI:", mymi, " reo: ", doreo, 'mdl:', mdlfn ) )
  return( list( c( qreg$fwdtransforms, initialTx ),  mymi ) )
  }


reoTemplate = antsImageRead( "templateImage.nii.gz" )
tefns = Sys.glob( 'bigdata/*rec.nii.gz' )
# testfn="bigdata/skull_157_11__rec.nii.gz"
evaldfFN = "evaldfB.csv"
if ( ! file.exists( evaldfFN ) ) {
  evaldf = data.frame( IDs=basename(tefns), MI=NA, MIB=NA  )
  } else evaldf = read.csv( evaldfFN )
for ( k in sample( 1:nrow( evaldf ) ) ) {
  print( k )
  if ( is.na( evaldf[k,'MIB'] ) ) {
    oimg = antsImageRead( tefns[k] )
    orinetmdlfn =  "models/mouse_rotation_3D_GPU_update1locsd5.h5"
    orinetmdlfnB =  "models/mouse_rotation_3D_GPU_update1locsd5b.h5"
    currentMI = Inf
    for ( rr in c(NULL,0,1,2) ) {
      transform = orinetInference( reoTemplate, oimg, orinetmdlfn, doreo=rr, newway=FALSE, verbose=T )
      transformB = orinetInference( reoTemplate, oimg, orinetmdlfnB, doreo=rr, newway=FALSE, verbose=T )
      if ( transformB[[2]] < transform[[2]]) transform=transformB
      if ( transform[[2]]  < currentMI ) {
        transformX = transform[[1]]
        currentMI = transform[[2]]
      }
    }
    mapped2gether = antsApplyTransforms( reoTemplate, oimg, transformX )
    evaldf[k,'MI']=antsImageMutualInformation( reoTemplate, mapped2gether )
    evaldf[k,'MAE']=mean( abs( reoTemplate - mapped2gether ) )
    print( evaldf[k,] )
    plot(  mapped2gether, axis=3, nslices=21, ncolumns=7, alpha=0.5 )
    write.csv( evaldf, evaldfFN, row.names=FALSE )
  }
}

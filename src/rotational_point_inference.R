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
    } else if ( doreo %in% c(0,1,2) ) {
      reo = reorientImage( oimg , axis1 = doreo )
      imglowreg = antsRegistration( reoTemplateOrinet, oimg, 'Translation', initialTransform=reo$txfn )
    } else { # random initialization
      randtx = randomAffineImage( oimg, transformType = "Rigid", sdAffine = 5, interpolation = "linear" )[[2]]
      imglowreg = antsRegistration( reoTemplateOrinet, oimg, 'Translation', initialTransform=randtx )
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
  mymi = antsImageMutualInformation( iMath(reoTemplateOrinet,"Normalize"), iMath(img2LM,"Normalize") )
  mymsq = mean( abs( iMath(reoTemplateOrinet,"Normalize") - iMath(img2LM,"Normalize") ) )
  if ( verbose )
    print( paste( "MI:", mymi, "MSQ:", mymsq, " reo: ", doreo, 'mdl:', mdlfn ) )
  return( list( c( qreg$fwdtransforms, initialTx ),  mymsq , mymi ) )
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
  if ( is.na( evaldf[k,'MI'] ) ) {
    oimg = antsImageRead( tefns[k] )
    orinetmdlfn =  "models/mouse_rotation_3D_GPU_update1locsd5.h5"
    orinetmdlfnB =  "models/mouse_rotation_3D_GPU_update1locsd5b.h5"
    currentMI = Inf
    currentMSQ = Inf
    for ( rr in c(NULL,0,1,2,3:6) ) {
      transform = orinetInference( reoTemplate, oimg, orinetmdlfn, doreo=rr, newway=FALSE, verbose=T )
      transformB = orinetInference( reoTemplate, oimg, orinetmdlfnB, doreo=rr, newway=FALSE, verbose=T )
      if ( transform[[2]]  < currentMSQ  &   transform[[3]]  < currentMI ) {
        transformX = transform[[1]]
        currentMSQ = transform[[2]]
        currentMI = transform[[3]]
      }
      if ( transformB[[2]]  < currentMSQ  &   transformB[[3]]  < currentMI ) {
        transformX = transformB[[1]]
        currentMSQ = transformB[[2]]
        currentMI = transformB[[3]]
      }
    }
    mapped2gether = antsApplyTransforms( reoTemplate, oimg, transformX )
    evaldf[k,'MI']=antsImageMutualInformation( reoTemplate, mapped2gether )
    evaldf[k,'MAE']=mean( abs( reoTemplate - mapped2gether ) )
    print( evaldf[k,] )
    pngfn = paste0( "transformationEvaluation/", tools::file_path_sans_ext( evaldf[k,'IDs'], T), "_reo.png" )
    plot(  mapped2gether, axis=3, nslices=21, ncolumns=7, alpha=0.5, outname=pngfn )
    write.csv( evaldf, evaldfFN, row.names=FALSE )
  }
}

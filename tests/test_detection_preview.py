"""Preview/production parity and visibility of uncomputed filtering stages."""
import numpy as np
import pandas as pd
import pytest
from firefly.analysis.fa_detection_preview import preview_detections
from firefly.analysis.fa_preprocess import filter_raw_contrast
from firefly.analysis.fa_roi import apply_roi_mask
from firefly.analysis import fa_localize


def movie():
    rng=np.random.default_rng(813)
    y,x=np.mgrid[:64,:64]
    frames=[]
    for i in range(3):
        img=rng.normal(200,3,(64,64))
        for cx,cy,a in ((20+i*.1,20,80),(44,43,12)):
            img+=a*np.exp(-((x-cx)**2+(y-cy)**2)/(2*1.3**2))
        frames.append(img)
    return np.array(frames,dtype=np.float32)


@pytest.mark.parametrize('backend',['trackpy','torch-cpu'])
@pytest.mark.parametrize('stream',[False,True])
def test_preview_matches_production_detections_and_contrast_roi_decisions(backend,stream,monkeypatch):
    if backend.startswith('torch'): pytest.importorskip('torch')
    raw=movie(); mask=np.ones(raw.shape[1:],bool);mask[:,35:]=False
    monkeypatch.setattr(fa_localize,'_ram_strategy',lambda *a,**k:(not stream,10.,.001))
    options=dict(diameter=7,minmass=.05,bg_radius=10,backend=backend,bg_method='uniform_filter')
    loc,*_=fa_localize.preprocess_and_localise_adaptive(raw,workers=1,chunk_size=2,**options)
    expected=apply_roi_mask(filter_raw_contrast(loc,raw,7,3),mask)
    for f in range(len(raw)):
        rows, summary=preview_detections(raw[f],roi_mask=mask,min_cnr=3,**options)
        actual=rows[rows.decision=='passes_detection'].sort_values(['x','y'])
        wanted=expected[expected.frame==f].sort_values(['x','y'])
        np.testing.assert_allclose(actual[['x','y','mass']],wanted[['x','y','mass']],rtol=1e-6,atol=1e-5)
        assert summary['candidates']==len(rows)
        assert summary['passed']+summary['contrast_rejected']+summary['outside_roi']==len(rows)
        assert (rows.mass >= .05).all()


def test_unknown_roi_is_not_reported_as_retained():
    rows, summary=preview_detections(movie()[0],backend='trackpy',roi_known=False)
    assert len(rows)>0 and rows.decision.eq('roi_unchecked').all()
    assert rows.inside_roi.isna().all() and not summary['roi_known']


def test_high_minmass_has_zero_candidates_not_a_stale_overlay():
    rows, summary=preview_detections(movie()[0],backend='trackpy',minmass=1e6)
    assert rows.empty and summary['candidates']==0 and summary['passed']==0


def test_exact_tiff_plane_and_unsupported_layout(tmp_path):
    import tifffile
    from firefly.ui.controllers.params.preview_loader import detection_frame
    raw=movie(); path=tmp_path/'movie.tif'
    tifffile.imwrite(path,raw,photometric='minisblack',metadata={'axes':'TYX'})
    np.testing.assert_array_equal(detection_frame(str(path),2),raw[2])
    path2=tmp_path/'channels.tif'
    tifffile.imwrite(path2,np.repeat(raw[:,None],2,axis=1),photometric='minisblack',metadata={'axes':'TCYX'})
    with pytest.raises(ValueError,match='unsupported'):
        detection_frame(str(path2),0)


def test_czi_preview_reads_the_selected_channel(monkeypatch):
    import sys,types
    from firefly.ui.controllers.params.preview_loader import detection_frame
    reads=[]
    class FakeCzi:
        dims='TCYX';size=(3,2,12,12)
        def __init__(self,path):pass
        def read_image(self,**kw):
            reads.append(kw)
            return np.full((1,1,12,12),kw['C']*10+kw['T']),None
    monkeypatch.setitem(sys.modules,'aicspylibczi',types.SimpleNamespace(CziFile=FakeCzi))
    assert np.all(detection_frame('movie.czi',2,channel=1)==12)
    assert reads==[{'T':2,'C':1}]

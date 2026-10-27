"""Small artificial fixtures exercise software behavior, not scientific accuracy."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from vesselviz.geometry import restore_box, crop_rectangle, clip_box
from vesselviz.privacy import mask_mmsi, short_name
from vesselviz.io import number, identifier, safe_key, confined_path, create_output
from vesselviz.render import association_panel, assemble_rows
from vesselviz.hull import hull_panels, polygon_mask
from vesselviz.association import collect_camera
from vesselviz.cli import run_association, run_hull, run_emissions, run_assemble
from vesselviz.emissions import POLLUTANTS, calculate, process_intervals, low_load_factor, validate_settings
from vesselviz.gallery import write_gallery


def parameters():
    # Deliberately simple test constants; not recommended emission parameters.
    return dict(draft_exponent=1., speed_exponent=3., load_cap=1., low_load_threshold=.2,
                load_points=[.01, .2], low_load_factors={p: [2., 1.] for p in POLLUTANTS},
                parameter_source="Artificial unit-test constants")


def interval():
    row = dict(area="A", clip="clip", passage_id="p1", vessel_id="test-vessel", timestamp="2000-01-01T00:00:00",
               power_kw=1000, speed_kn=5, design_speed_kn=10, design_draft_m=4,
               visual_draft_m=2, dt_s=1, vessel_type="test")
    row.update({"ef_"+p+"_g_kwh": 1 for p in POLLUTANTS})
    return row


class GeometryTests(unittest.TestCase):
    def test_identity(self): self.assertEqual(restore_box([1, 2, 3, 4], [100, 80], {"space": "original"}), [1, 2, 3, 4])
    def test_crop_resize(self):
        spec = dict(space="crop_resize", crop=[10, 20, 5, 15], resized_size=[80, 50])
        self.assertEqual(restore_box([0, 0, 80, 50], [100, 80], spec), [5, 10, 85, 60])
    def test_invalid_crop(self):
        for crop in ([None]*4, [0, 100, 0, 0], [-1, 0, 0, 0]):
            with self.assertRaises(ValueError): crop_rectangle([100, 80], crop)
    def test_unknown_space(self):
        with self.assertRaises(ValueError): restore_box([1, 2, 3, 4], [100, 80], {})
    def test_clipping(self): self.assertEqual(clip_box([-2, 1, 5, 8], [0, 0, 9, 9]), [0, 1, 5, 8])
    def test_outside(self): self.assertIsNone(clip_box([10, 10, 20, 20], [0, 0, 5, 5]))


class SafetyTests(unittest.TestCase):
    def test_identifiers(self):
        self.assertEqual(identifier("12.0"), "12"); self.assertEqual(identifier("nan"), "")
        self.assertIsNone(number("inf")); self.assertIsNone(number("nan"))
    def test_mmsi(self):
        self.assertEqual(mask_mmsi("000012345"), "0000*****")
        with self.assertRaises(ValueError): mask_mmsi("123")
    def test_names(self):
        self.assertEqual(short_name("Sample Ship"), "SA**"); self.assertEqual(short_name("0"), "N/A")
        self.assertEqual(short_name("测试"), "N/A")
    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ("../outside", "..\\outside", "/tmp/escape", "C:/escape"):
                with self.assertRaises(ValueError): confined_path(root, name)
    def test_key_escape(self):
        for key in ("../bad", "", "A/B", "CON", "nul.txt", "com1", "frame."):
            with self.assertRaises(ValueError): safe_key(key)
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileExistsError): create_output(root)
    def test_gallery_escape(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"gallery.html"; write_gallery(p,[dict(id="a",frame="</script><script>alert(1)</script>")])
            self.assertNotIn("</script><script>alert",p.read_text())


class RenderTests(unittest.TestCase):
    def test_panel_fields_and_source(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"frame.png"; Image.new("RGB",(100,80),"gray").save(p); before=p.read_bytes()
            ship=dict(track_id="12",bbox=[10,20,50,60],mmsi_masked="0000*****",speed_text="1.0 kn",name_masked="SA**")
            panel,lo=association_panel(dict(image_path=str(p),size=[100,80],display_crop=[2,3,0,0],ships=[ship]))
            self.assertEqual(panel.width,1600);self.assertEqual(p.read_bytes(),before)
            self.assertEqual(lo['cards'][0][0]['y'],lo['cards'][0][1]['y'])
            self.assertEqual(lo['cards'][0][1]['x'],lo['cards'][0][3]['x'])
    def test_assembly(self):
        a=Image.new("RGB",(20,10));b=Image.new("RGB",(20,20))
        self.assertEqual(assemble_rows([[a,b]],100,8).size,(208,100))
    def test_polygon_rejection(self):
        with self.assertRaises(ValueError):polygon_mask([[0,0],[1,1],[2,2]],(10,10))


class EmissionTests(unittest.TestCase):
    def test_energy_units(self):
        args=dict(power_kw=[1000],speed_kn=[10],design_speed_kn=[10],draft_m=[4],design_draft_m=[4],dt_s=[3600],
                  factors_g_kwh={p:[1] for p in POLLUTANTS},settings=parameters())
        load,emissions=calculate(**args)
        self.assertEqual(load[0],1);self.assertEqual(emissions['CO2'][0],1000)
    def test_draft_equal(self):
        row=interval();row['visual_draft_m']=row['design_draft_m']; detail,summary,_=process_intervals([row],parameters())
        self.assertEqual(summary[0]['emission_kg'],summary[6]['emission_kg'])
    def test_zero_speed(self):
        row=interval();row['speed_kn']=0
        self.assertTrue(all(r['emission_kg']==0 for r in process_intervals([row],parameters())[1]))
    def test_duration(self):
        a=interval();b=dict(a,dt_s=2)
        sa=process_intervals([a],parameters())[1];sb=process_intervals([b],parameters())[1]
        self.assertTrue(all(abs(y['emission_kg']-2*x['emission_kg'])<1e-12 for x,y in zip(sa,sb)))
    def test_llaf(self):
        p=parameters();np.testing.assert_allclose(low_load_factor([0,.01,.105,.2,.5],'CO2',p),[1,2,1.5,1,1])
    def test_missing_parameters(self):
        with self.assertRaises(ValueError):validate_settings({})
    def test_invalid_row_logged(self):
        a=interval();b=dict(a,timestamp="2000-01-01T00:00:01",visual_draft_m="")
        _,_,issues=process_intervals([a,b],parameters());self.assertEqual(len(issues),1)
    def test_duplicate_logged(self):
        a=interval();_,_,issues=process_intervals([a,a],parameters());self.assertEqual(len(issues),1)
    def test_reference_requires_source(self):
        a=dict(interval(),reference_draft_m=3)
        with self.assertRaises(ValueError):process_intervals([a],parameters(),True)
    def test_reference_same_intervals(self):
        a=dict(interval(),reference_draft_m=3,reference_source="test",reference_kind="test-only")
        detail,_,_=process_intervals([a],parameters(),True);self.assertEqual(len(detail),18)
    def test_overlap_between_passages(self):
        a=interval();b=dict(a,passage_id="p2")
        with self.assertRaises(ValueError):process_intervals([a,b],parameters())
    def test_conflicting_vessels(self):
        a=interval();b=dict(a,vessel_id="different",timestamp="2000-01-01T00:00:01")
        with self.assertRaises(ValueError):process_intervals([a,b],parameters())
    def test_same_timestamp_conflicting_vessels(self):
        a=interval();b=dict(a,vessel_id="different")
        with self.assertRaises(ValueError):process_intervals([a,b],parameters())
    def test_conflicting_types(self):
        a=interval();b=dict(a,vessel_type="different",timestamp="2000-01-01T00:00:01")
        with self.assertRaises(ValueError):process_intervals([a,b],parameters())
    def test_parameter_types(self):
        for value in (True, "1"):
            p=parameters();p['load_cap']=value
            with self.assertRaises(ValueError):validate_settings(p)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def json(self,name,value):
        p=self.root/name;p.write_text(json.dumps(value));return p
    def csv(self,name,rows):
        p=self.root/name
        with p.open('w',newline='') as s:
            w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return p
    def association_fixture(self):
        Image.new('RGB',(100,80),'gray').save(self.root/'frame.png')
        row=dict(image_name='frame.png',track_id='1',ais_mmsi='000012345',x1=10,y1=20,x2=40,y2=50,
                 ais_speed=2,ais_english_name='Test Boat',match_status='MATCHED',track_match_confidence='high',within_adaptive_gate='True')
        self.csv('matches.csv',[row]);self.csv('mapping.csv',[dict(track_id='1',mmsi='000012345')])
        camera=dict(camera='C1',clip='clip',matches_csv='matches.csv',mapping_csv='mapping.csv',images_dir='.',
                    original_size=[100,80],display_crop=[0,0,0,0],coordinates={'space':'original'},speed_unit='kn',accepted_statuses=['MATCHED'])
        return camera,row
    def test_association_cli(self):
        cam,_=self.association_fixture();cfg=self.json('config.json',dict(cameras=[cam]))
        out=run_association(cfg,self.root/'out');index=json.loads((out/'index.json').read_text())
        selection=self.json('selection.json',dict(ids=[index[0]['id']]))
        assembled=run_assemble(out/'index.json',selection,self.root/'grid')
        self.assertTrue((assembled/'selected_grid.png').exists())
        self.assertNotIn('000012345',(out/'index.json').read_text())
    def test_duplicate_association_excluded(self):
        cam,row=self.association_fixture();self.csv('matches.csv',[row,row]);frames,issues=collect_camera(cam,self.root)
        self.assertEqual(frames,[]);self.assertEqual(sum(i['code']=='duplicate_track_excluded' for i in issues),2)
    def test_mapping_conflict(self):
        cam,_=self.association_fixture();self.csv('mapping.csv',[dict(track_id='1',mmsi='000012345'),dict(track_id='1',mmsi='000012346')])
        frames,issues=collect_camera(cam,self.root);self.assertEqual(frames,[]);self.assertEqual(issues[0]['code'],'mapping_conflict')
    def test_missing_image(self):
        cam,row=self.association_fixture();row['image_name']='missing.png';self.csv('matches.csv',[row])
        frames,issues=collect_camera(cam,self.root);self.assertEqual(frames,[]);self.assertEqual(issues[-1]['code'],'missing_or_invalid_image')
    def hull_fixture(self):
        Image.new('RGB',(20,10),'gray').save(self.root/'source.png')
        Image.new('L',(20,10),0).save(self.root/'mask.png')
        np.save(self.root/'prob.npy',np.full((10,20),.25,dtype='float32'))
        self.json('hulls.json',dict(hull_instances=[dict(id='h1',polygon=[[2,2],[9,2],[9,6],[2,6]])]))
        return dict(camera='C1',clip='clip',frame_id='frame',image='source.png',original_size=[20,10],
                    vessel_mask='mask.png',probability='prob.npy',hulls='hulls.json',display_crop=[0,0,0,0])
    def test_hull_cli(self):
        f=self.hull_fixture();m=self.json('manifest.json',dict(frames=[f]));out=run_hull(m,self.root/'hull-output')
        self.assertTrue((out/'all_frames.png').exists())
    def test_hull_outside_unchanged(self):
        f=self.hull_fixture();panels,_=hull_panels(f,self.root)
        self.assertEqual(panels[0].getpixel((0,0)),panels[4].getpixel((0,0)))
    def test_missing_identity_warning(self):
        _,report=hull_panels(self.hull_fixture(),self.root)
        self.assertTrue(any('metadata missing' in s for s in report['warnings']))
    def test_bad_probability(self):
        f=self.hull_fixture();np.save(self.root/'prob.npy',np.full((10,20),2.))
        with self.assertRaises(ValueError):hull_panels(f,self.root)
    def test_missing_hull_input(self):
        f=self.hull_fixture();f['hulls']='missing.json'
        with self.assertRaises(FileNotFoundError):hull_panels(f,self.root)
    def test_emissions_cli(self):
        self.csv('inputs.csv',[interval()]);p=self.json('emissions.json',dict(input_csv='inputs.csv',model=parameters()))
        out=run_emissions(p,self.root/'emissions-output')
        self.assertTrue((out/'passage_comparison.csv').exists());self.assertTrue((out/'passage_rankings.csv').exists())


if __name__=='__main__':unittest.main()

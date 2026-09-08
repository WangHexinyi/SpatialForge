// SpatialForge God View Research Inspector — G1.3 Frontend Application
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

// Enforce Z-up coordinate system matching SpatialForge geometry
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const COLOR_MAP = {
  red: 0xcc1414,
  blue: 0x1428cc,
  green: 0x149914,
  yellow: 0xe6cc14,
  purple: 0x8014b3,
  cyan: 0x14b3b3,
};

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

class GodViewApp {
  constructor() {
    this.canvas = document.getElementById('godview-canvas');
    this.mode = 'sample'; // 'sample' (Inspect Sample) or 'explore' (Explore Scene)
    this.currentSnapshot = null;
    this.currentRunId = null;
    this.currentSceneId = null;
    this.currentViewId = null;
    this.currentSampleId = null;
    this.selectedObjectIndex = null;
    this.activeTab = 'sample'; // 'sample' or 'object'

    this.allSamples = [];
    this.filteredSamples = [];
    this.activeSearchIndex = -1;

    this.profiles = [];
    this.runs = [];
    this.scenes = [];
    this.views = [];

    this.currentTrajectory = null;
    this.trajectoryFrameIndex = 0;
    this.isTrajectoryPlaying = false;
    this.playbackAnimFrame = null;
    this.playbackElapsedSec = 0.0;
    this.lastPlaybackTime = null;
    this.previewPlaybackRate = 1.0;
    this.scientificSpeed = 1.0;
    this.trajectoryFamily = 'survey_orbit';

    this.currentViewSetId = 'historical_8';
    this.viewSetsData = null;
    this.currentExploreDistanceMode = 'historical_fixed';
    this.currentManualDistance = 6.0;
    this.currentDistanceScale = 1.0;
    this.showCanonicalCams = false;

    this.waypointInterpMode = 'catmull_rom';
    this.waypoints = [
      { x: 3.5, y: -3.5, z: 2.2, tx: 0.0, ty: 0.0, tz: 0.4 },
      { x: 0.0, y: -4.5, z: 2.8, tx: 0.0, ty: 0.0, tz: 0.4 },
      { x: -3.5, y: -3.5, z: 2.2, tx: 0.0, ty: 0.0, tz: 0.4 },
      { x: 0.0, y: -2.0, z: 1.5, tx: 0.0, ty: 0.0, tz: 0.4 },
    ];

    this.previewCanvas = document.getElementById('traj-preview-canvas');
    this.previewRenderer = null;
    this.previewCamera = null;

    this.init3D();
    this.bindEvents();
    this.bootstrap();
    this.startTelemetryPolling();
  }

  init3D() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x090b0e);

    const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 100);
    this.camera.position.set(0, -10, 6);
    this.camera.lookAt(0, 0, 0.4);

    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
    this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, 0.4);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    // Grid in XY plane (Z=0) with subtle neutral lines
    const grid = new THREE.GridHelper(12, 24, 0x282e35, 0x171b20);
    grid.rotation.x = Math.PI / 2;
    this.scene.add(grid);

    // World coordinate axes: Red=X, Green=Y, Blue=Z
    const axes = new THREE.AxesHelper(2.5);
    this.scene.add(axes);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight.position.set(5, -8, 12);
    this.scene.add(dirLight);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
    dirLight2.position.set(-6, 8, 6);
    this.scene.add(dirLight2);

    // Groups for dynamic assets
    this.objectsGroup = new THREE.Group();
    this.cameraMarkerGroup = new THREE.Group();
    this.highlightGroup = new THREE.Group();
    this.trajectoryGroup = new THREE.Group();
    this.supportLineGroup = new THREE.Group();
    this.showSupportLines = true;
    this.canonicalCamsGroup = new THREE.Group();
    this.waypointMarkersGroup = new THREE.Group();
    this.scene.add(this.objectsGroup);
    this.scene.add(this.cameraMarkerGroup);
    this.scene.add(this.highlightGroup);
    this.scene.add(this.trajectoryGroup);
    this.scene.add(this.supportLineGroup);
    this.scene.add(this.canonicalCamsGroup);
    this.scene.add(this.waypointMarkersGroup);

    // Live Observation Preview Camera and Renderer (Fast WebGL browser preview)
    if (this.previewCanvas) {
      this.previewRenderer = new THREE.WebGLRenderer({ canvas: this.previewCanvas, antialias: true });
      this.previewRenderer.setSize(320, 320);
      this.previewRenderer.setPixelRatio(window.devicePixelRatio || 1);
      this.previewCamera = new THREE.PerspectiveCamera(60, 1.0, 0.1, 100);
    }

    // Raycaster for object clicking
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);

      // Render live preview if in trajectory mode
      if (this.mode === 'trajectory' && this.currentTrajectory && this.previewRenderer && this.previewCamera) {
        const trFrame = this.currentTrajectory.frames[this.trajectoryFrameIndex];
        if (trFrame) {
          this.updateObservationPreview(trFrame.pose);
        }
      }

      // Update HUD camera position
      const cp = this.camera.position;
      const camInfoEl = document.getElementById('hud-cam-info');
      if (camInfoEl) {
        camInfoEl.textContent = `God Cam: (${cp.x.toFixed(1)}, ${cp.y.toFixed(1)}, ${cp.z.toFixed(1)})`;
      }
    };
    animate();

    this.handleResize();
    window.addEventListener('resize', () => {
      this.handleResize();
    });
    if (window.ResizeObserver && this.canvas && this.canvas.parentElement) {
      const ro = new ResizeObserver(() => {
        this.handleResize();
      });
      ro.observe(this.canvas.parentElement);
    }
  }

  handleResize() {
    if (!this.canvas || !this.canvas.parentElement) return;
    const w = this.canvas.parentElement.clientWidth;
    const h = this.canvas.parentElement.clientHeight;
    if (w > 0 && h > 0 && this.camera && this.renderer) {
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    }
  }

  bindEvents() {
    // 1. Mode Switcher
    document.getElementById('btn-mode-sample').addEventListener('click', () => {
      this.setMode('sample');
    });

    document.getElementById('btn-mode-explore').addEventListener('click', () => {
      this.setMode('explore');
    });

    const btnModeTraj = document.getElementById('btn-mode-trajectory');
    if (btnModeTraj) {
      btnModeTraj.addEventListener('click', () => {
        this.setMode('trajectory');
      });
    }

    // 2. Inspector Tabs (Sample | Trajectory | Object)
    document.getElementById('tab-btn-sample').addEventListener('click', () => {
      this.setActiveTab('sample');
    });

    const tabBtnTraj = document.getElementById('tab-btn-trajectory');
    if (tabBtnTraj) {
      tabBtnTraj.addEventListener('click', () => {
        this.setActiveTab('trajectory');
      });
    }

    document.getElementById('tab-btn-object').addEventListener('click', () => {
      this.setActiveTab('object');
    });

    // 2b. Trajectory Lab Controls
    const trajSceneSel = document.getElementById('traj-scene-select');
    if (trajSceneSel) {
      trajSceneSel.addEventListener('change', (e) => {
        this.currentSceneId = e.target.value;
        this.loadTrajectory();
      });
    }

    const trajFamilySel = document.getElementById('traj-family-select');
    const waypointEditorSec = document.getElementById('section-waypoint-editor');
    if (trajFamilySel) {
      trajFamilySel.addEventListener('change', (e) => {
        this.trajectoryFamily = e.target.value;
        if (waypointEditorSec) {
          waypointEditorSec.style.display = (this.trajectoryFamily === 'custom_waypoints') ? 'block' : 'none';
        }
        this.renderWaypoint3D();
        this.loadTrajectory();
      });
    }

    const trajFramingSel = document.getElementById('traj-framing-select');
    const trajManualWrap = document.getElementById('traj-manual-dist-wrap');
    const trajScaleWrap = document.getElementById('traj-scale-wrap');
    if (trajFramingSel) {
      trajFramingSel.addEventListener('change', (e) => {
        const val = e.target.value;
        if (trajManualWrap) trajManualWrap.style.display = (val === 'manual') ? 'flex' : 'none';
        if (trajScaleWrap) trajScaleWrap.style.display = (val === 'adaptive_scale') ? 'flex' : 'none';
        this.loadTrajectory();
      });
    }

    const trajDistInput = document.getElementById('traj-dist-input');
    const trajDistSlider = document.getElementById('traj-dist-slider');
    if (trajDistInput) {
      trajDistInput.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v >= 0.5 && v <= 20.0) {
          if (trajDistSlider) trajDistSlider.value = v;
        }
      });
      trajDistInput.addEventListener('change', () => {
        this.loadTrajectory();
      });
    }
    if (trajDistSlider) {
      trajDistSlider.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        if (trajDistInput) trajDistInput.value = v.toFixed(1);
      });
      trajDistSlider.addEventListener('change', () => {
        this.loadTrajectory();
      });
    }

    const trajSpeedInput = document.getElementById('traj-speed-input');
    if (trajSpeedInput) {
      trajSpeedInput.addEventListener('change', () => {
        this.loadTrajectory();
      });
    }

    const trajRateSel = document.getElementById('traj-rate-select');
    if (trajRateSel) {
      trajRateSel.addEventListener('change', (e) => {
        this.previewPlaybackRate = parseFloat(e.target.value) || 1.0;
      });
    }

    const trajFramesSel = document.getElementById('traj-frames-select');
    if (trajFramesSel) {
      trajFramesSel.addEventListener('change', () => {
        this.loadTrajectory();
      });
    }

    const trajSeedInput = document.getElementById('traj-seed-input');
    if (trajSeedInput) {
      trajSeedInput.addEventListener('change', () => {
        this.loadTrajectory();
      });
    }

    const btnTrajGen = document.getElementById('btn-traj-generate');
    if (btnTrajGen) {
      btnTrajGen.addEventListener('click', () => {
        this.loadTrajectory();
      });
    }

    const btnTrajPlay = document.getElementById('btn-traj-play');
    if (btnTrajPlay) {
      btnTrajPlay.addEventListener('click', () => {
        this.toggleTrajectoryPlayback();
      });
    }

    const btnTrajReset = document.getElementById('btn-traj-reset');
    if (btnTrajReset) {
      btnTrajReset.addEventListener('click', () => {
        this.seekTrajectoryFrame(0, true);
      });
    }

    const trajSlider = document.getElementById('traj-timeline-slider');
    if (trajSlider) {
      trajSlider.addEventListener('input', (e) => {
        this.seekTrajectoryFrame(parseInt(e.target.value, 10), true);
      });
    }

    const btnWaypointsToggle = document.getElementById('btn-traj-waypoints-toggle');
    if (btnWaypointsToggle && waypointEditorSec) {
      btnWaypointsToggle.addEventListener('click', () => {
        const isHidden = waypointEditorSec.style.display === 'none';
        waypointEditorSec.style.display = isHidden ? 'block' : 'none';
        if (isHidden && trajFamilySel && trajFamilySel.value !== 'custom_waypoints') {
          trajFamilySel.value = 'custom_waypoints';
          this.trajectoryFamily = 'custom_waypoints';
          this.renderWaypoint3D();
          this.loadTrajectory();
        }
      });
    }

    // Custom Waypoint Editor controls
    const btnWpAdd = document.getElementById('btn-waypoint-add');
    if (btnWpAdd) {
      btnWpAdd.addEventListener('click', () => {
        this.addWaypoint();
      });
    }

    const btnWpTogglePaste = document.getElementById('btn-waypoint-toggle-paste');
    const wpPasteBox = document.getElementById('waypoint-paste-box');
    const btnWpClosePaste = document.getElementById('btn-waypoint-close-paste');
    if (btnWpTogglePaste && wpPasteBox) {
      btnWpTogglePaste.addEventListener('click', () => {
        wpPasteBox.style.display = (wpPasteBox.style.display === 'none') ? 'flex' : 'none';
      });
    }
    if (btnWpClosePaste && wpPasteBox) {
      btnWpClosePaste.addEventListener('click', () => {
        wpPasteBox.style.display = 'none';
      });
    }

    const btnWpApplyPaste = document.getElementById('btn-waypoint-apply-paste');
    if (btnWpApplyPaste) {
      btnWpApplyPaste.addEventListener('click', () => {
        this.applyWaypointPaste();
      });
    }

    const btnWpExport = document.getElementById('btn-waypoint-export-text');
    if (btnWpExport) {
      btnWpExport.addEventListener('click', () => {
        this.exportWaypointsToText();
      });
    }

    const btnWpExample = document.getElementById('btn-waypoint-example');
    if (btnWpExample) {
      btnWpExample.addEventListener('click', () => {
        this.loadDefaultWaypoints();
      });
    }

    const btnWpClear = document.getElementById('btn-waypoint-clear');
    if (btnWpClear) {
      btnWpClear.addEventListener('click', () => {
        this.clearWaypoints();
      });
    }

    const wpInterpSel = document.getElementById('waypoint-interp-mode');
    if (wpInterpSel) {
      wpInterpSel.addEventListener('change', (e) => {
        this.waypointInterpMode = e.target.value;
        this.renderWaypoint3D();
        this.loadTrajectory();
      });
    }

    const btnWpSync = document.getElementById('btn-waypoint-sync-generate');
    if (btnWpSync) {
      btnWpSync.addEventListener('click', () => {
        this.loadTrajectory();
      });
    }

    const btnAdvToggle = document.getElementById('btn-traj-adv-toggle');
    const advDrawer = document.getElementById('traj-adv-drawer');
    const btnAdvClose = document.getElementById('btn-traj-adv-close');
    if (btnAdvToggle && advDrawer) {
      btnAdvToggle.addEventListener('click', () => {
        const isHidden = advDrawer.style.display === 'none';
        advDrawer.style.display = isHidden ? 'block' : 'none';
      });
    }
    if (btnAdvClose && advDrawer) {
      btnAdvClose.addEventListener('click', () => {
        advDrawer.style.display = 'none';
      });
    }

    // 3. Experiment Context Drawer Toggle & Close
    const btnExpDrawer = document.getElementById('btn-experiment-drawer');
    btnExpDrawer.addEventListener('click', () => {
      this.toggleExperimentDrawer(true);
    });

    document.getElementById('btn-close-drawer').addEventListener('click', () => {
      this.toggleExperimentDrawer(false);
    });

    document.getElementById('drawer-backdrop').addEventListener('click', () => {
      this.toggleExperimentDrawer(false);
    });

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        this.toggleExperimentDrawer(false);
        this.closeSampleDropdown();
      }
    });

    // 4. Sample Search & Dropdown Navigation
    const searchInput = document.getElementById('sample-search-input');
    const dropdown = document.getElementById('sample-dropdown');

    searchInput.addEventListener('focus', () => {
      this.filterSampleDropdown(searchInput.value);
    });

    searchInput.addEventListener('input', (e) => {
      this.filterSampleDropdown(e.target.value);
    });

    searchInput.addEventListener('keydown', (e) => {
      if (dropdown.style.display === 'none') {
        if (e.key === 'ArrowDown' || e.key === 'Enter') {
          this.filterSampleDropdown(searchInput.value);
        }
        return;
      }

      const items = dropdown.querySelectorAll('.sample-opt');
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        this.activeSearchIndex = Math.min(this.activeSearchIndex + 1, items.length - 1);
        this.highlightDropdownItem(items);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        this.activeSearchIndex = Math.max(this.activeSearchIndex - 1, 0);
        this.highlightDropdownItem(items);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (this.activeSearchIndex >= 0 && this.activeSearchIndex < items.length) {
          const sampleId = items[this.activeSearchIndex].getAttribute('data-id');
          this.selectSampleById(sampleId);
        } else if (items.length > 0) {
          const sampleId = items[0].getAttribute('data-id');
          this.selectSampleById(sampleId);
        }
      } else if (e.key === 'Escape') {
        this.closeSampleDropdown();
      }
    });

    // Stepper buttons for samples
    document.getElementById('btn-sample-prev').addEventListener('click', () => {
      this.stepSample(-1);
    });
    document.getElementById('btn-sample-next').addEventListener('click', () => {
      this.stepSample(1);
    });

    // Close sample dropdown if clicked outside
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.sample-search-field')) {
        this.closeSampleDropdown();
      }
    });

    // 5. Explore Mode Controls (View Sets, Views & Distance)
    const viewSetSel = document.getElementById('viewset-select');
    if (viewSetSel) {
      viewSetSel.addEventListener('change', (e) => {
        this.onViewSetChange(e.target.value);
      });
    }

    const btnViewPrev = document.getElementById('btn-view-prev');
    if (btnViewPrev) {
      btnViewPrev.addEventListener('click', () => {
        this.stepView(-1);
      });
    }

    const btnViewNext = document.getElementById('btn-view-next');
    if (btnViewNext) {
      btnViewNext.addEventListener('click', () => {
        this.stepView(1);
      });
    }

    document.getElementById('scene-select').addEventListener('change', async (e) => {
      this.currentSceneId = e.target.value;
      await this.loadViewSets(this.currentSceneId);
      this.loadSnapshot();
    });

    document.getElementById('view-select').addEventListener('change', (e) => {
      this.currentViewId = e.target.value;
      this.updateViewStepperUI();
      if (this.showCanonicalCams) {
        this.renderCanonicalCameraMarkers();
      }
      this.loadSnapshot();
    });

    // Distance controls for Explore Scene mode
    const expDistModeSel = document.getElementById('explore-dist-mode');
    const expManualWrap = document.getElementById('explore-manual-dist-wrap');
    const expScaleWrap = document.getElementById('explore-scale-wrap');
    const expDistInput = document.getElementById('explore-dist-input');
    const expDistSlider = document.getElementById('explore-dist-slider');
    const expScaleSel = document.getElementById('explore-scale-select');

    if (expDistModeSel) {
      expDistModeSel.addEventListener('change', (e) => {
        const m = e.target.value;
        this.currentExploreDistanceMode = m;
        if (expManualWrap) expManualWrap.style.display = (m === 'manual') ? 'flex' : 'none';
        if (expScaleWrap) expScaleWrap.style.display = (m === 'adaptive_scale') ? 'flex' : 'none';
        this.loadSnapshot();
      });
    }

    if (expDistInput) {
      expDistInput.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v >= 0.5 && v <= 20.0) {
          this.currentManualDistance = v;
          if (expDistSlider) expDistSlider.value = v;
          this.applyCameraDistance(v);
        }
      });
      expDistInput.addEventListener('change', (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v >= 0.5 && v <= 20.0) {
          this.currentManualDistance = v;
          if (expDistSlider) expDistSlider.value = v;
          this.loadSnapshot();
        }
      });
    }

    if (expDistSlider) {
      expDistSlider.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        this.currentManualDistance = v;
        if (expDistInput) expDistInput.value = v.toFixed(1);
        this.applyCameraDistance(v);
      });
      expDistSlider.addEventListener('change', (e) => {
        const v = parseFloat(e.target.value);
        this.currentManualDistance = v;
        if (expDistInput) expDistInput.value = v.toFixed(1);
        this.loadSnapshot();
      });
    }

    if (expScaleSel) {
      expScaleSel.addEventListener('change', (e) => {
        this.currentDistanceScale = parseFloat(e.target.value);
        this.loadSnapshot();
      });
    }

    // 6. Run Selection in Drawer
    document.getElementById('run-select').addEventListener('change', (e) => {
      this.currentRunId = e.target.value || null;
      this.updateExperimentContextButton();
      this.updateDrawerRunDetails();
      this.loadSnapshot();
    });

    // 7. Profile Selection in Drawer
    document.getElementById('profile-select').addEventListener('change', (e) => {
      this.updateDrawerProfileDetails(e.target.value);
    });

    // Profile Preflight check button in drawer
    document.getElementById('btn-preflight').addEventListener('click', () => {
      const profId = document.getElementById('profile-select').value;
      this.validateProfilePreflight(profId);
    });

    // 8. 3D Camera Controls
    document.getElementById('btn-reset-cam').addEventListener('click', () => {
      this.camera.position.set(0, -10, 6);
      this.controls.target.set(0, 0, 0.4);
      this.camera.lookAt(0, 0, 0.4);
    });

    document.getElementById('btn-align-obs').addEventListener('click', () => {
      if (!this.currentSnapshot) return;
      const pose = this.currentSnapshot.camera.pose;
      this.camera.position.set(...pose.position);
      this.controls.target.set(...pose.look_at);
      this.camera.lookAt(new THREE.Vector3(...pose.look_at));
    });

    const btnToggleSup = document.getElementById('btn-toggle-support');
    if (btnToggleSup) {
      btnToggleSup.addEventListener('click', () => {
        this.showSupportLines = !this.showSupportLines;
        btnToggleSup.textContent = this.showSupportLines ? 'Support: On' : 'Support: Off';
        if (this.currentSnapshot && this.currentSnapshot.environment_truth) {
          this.updateSupportLines(this.currentSnapshot.environment_truth.objects, this.selectedObjectIndex);
        }
      });
    }

    const btnToggleCanon = document.getElementById('btn-toggle-canonical-cams');
    if (btnToggleCanon) {
      btnToggleCanon.addEventListener('click', () => {
        this.toggleCanonicalCameras();
      });
    }

    // 9. Raycasting Click Selection
    this.canvas.addEventListener('pointerdown', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);

      // Check if a canonical camera was clicked in God View
      if (this.showCanonicalCams && this.canonicalCamsGroup.children.length > 0) {
        const camHits = this.raycaster.intersectObjects(this.canonicalCamsGroup.children, true);
        if (camHits.length > 0) {
          let hit = camHits[0].object;
          if (hit && hit.userData && hit.userData.view_id) {
            this.currentViewId = hit.userData.view_id;
            const sel = document.getElementById('view-select');
            if (sel) sel.value = this.currentViewId;
            this.updateViewStepperUI();
            this.renderCanonicalCameraMarkers();
            this.loadSnapshot();
            return;
          }
        }
      }

      const intersects = this.raycaster.intersectObjects(this.objectsGroup.children, true);
      if (intersects.length > 0) {
        let hit = intersects[0].object;
        while (hit && hit.userData.object_index === undefined && hit.parent) {
          hit = hit.parent;
        }
        if (hit && hit.userData.object_index !== undefined) {
          this.selectObject(hit.userData.object_index);
        }
      }
    });
  }

  async bootstrap() {
    await this.loadProfiles();
    await this.loadRuns();
    await this.loadAllSamples();
    await this.loadScenes();
    this.initWaypointEditor();

    // Default mode: Inspect Sample
    this.setMode('sample');
  }

  // =========================================================================
  // Mode Management
  // =========================================================================
  async setMode(mode) {
    this.mode = mode;
    const btnSample = document.getElementById('btn-mode-sample');
    const btnExplore = document.getElementById('btn-mode-explore');
    const btnTraj = document.getElementById('btn-mode-trajectory');

    const ctrlSample = document.getElementById('mode-controls-sample');
    const ctrlExplore = document.getElementById('mode-controls-explore');
    const ctrlTraj = document.getElementById('mode-controls-trajectory');
    const trajToolbarRow = document.getElementById('traj-toolbar-row');

    const secQ = document.getElementById('section-question');
    const secP = document.getElementById('section-prediction');
    const secT = document.getElementById('section-spatial-truth');
    const notice = document.getElementById('explore-sample-notice');
    const chSection = document.getElementById('section-challenge-summary');

    const tabBtnSample = document.getElementById('tab-btn-sample');
    const tabBtnTraj = document.getElementById('tab-btn-trajectory');

    // Reset playback clock & pause when switching modes
    if (mode !== 'trajectory') {
      if (this.isTrajectoryPlaying) {
        this.pauseTrajectoryPlayback();
      }
      this.resetTrajectoryPlaybackClock();
      if (this.waypointMarkersGroup) {
        this.waypointMarkersGroup.visible = false;
      }
    }

    if (mode === 'sample') {
      btnSample.classList.add('active');
      btnExplore.classList.remove('active');
      if (btnTraj) btnTraj.classList.remove('active');

      ctrlSample.style.display = 'flex';
      ctrlExplore.style.display = 'none';
      if (ctrlTraj) ctrlTraj.style.display = 'none';
      if (trajToolbarRow) trajToolbarRow.style.display = 'none';

      secQ.style.display = 'flex';
      secP.style.display = 'flex';
      secT.style.display = 'flex';
      notice.style.display = 'none';
      if (chSection) chSection.style.display = 'none';

      if (tabBtnSample) tabBtnSample.style.display = 'inline-block';
      if (tabBtnTraj) tabBtnTraj.style.display = 'none';
      this.setActiveTab('sample');

      this.trajectoryGroup.visible = false;

      // Ensure a sample is active
      if (!this.currentSampleId && this.allSamples.length > 0) {
        this.selectSample(this.allSamples[0]);
      } else if (this.currentSampleId) {
        const s = this.allSamples.find((item) => (item.id || item.sample_id) === this.currentSampleId);
        if (s) this.selectSample(s);
        else this.loadSnapshot();
      }
    } else if (mode === 'explore') {
      btnSample.classList.remove('active');
      btnExplore.classList.add('active');
      if (btnTraj) btnTraj.classList.remove('active');

      ctrlSample.style.display = 'none';
      ctrlExplore.style.display = 'flex';
      if (ctrlTraj) ctrlTraj.style.display = 'none';
      if (trajToolbarRow) trajToolbarRow.style.display = 'none';

      secQ.style.display = 'none';
      secP.style.display = 'none';
      secT.style.display = 'none';
      notice.style.display = 'flex';
      if (chSection) chSection.style.display = 'flex';

      if (tabBtnSample) tabBtnSample.style.display = 'inline-block';
      if (tabBtnTraj) tabBtnTraj.style.display = 'none';
      this.setActiveTab('sample');

      this.trajectoryGroup.visible = false;

      // Ensure scenes and views are ready
      if (!this.currentSceneId && this.scenes.length > 0) {
        this.currentSceneId = this.scenes[0];
      }
      document.getElementById('scene-select').value = this.currentSceneId || '';

      if (!this.viewSetsData) {
        await this.loadViewSets(this.currentSceneId);
      } else {
        this.applyActiveViewSet();
      }

      // Clear QA sample
      this.currentSampleId = null;
      this.loadSnapshot();
    } else if (mode === 'trajectory') {
      btnSample.classList.remove('active');
      btnExplore.classList.remove('active');
      if (btnTraj) btnTraj.classList.add('active');

      ctrlSample.style.display = 'none';
      ctrlExplore.style.display = 'none';
      if (ctrlTraj) ctrlTraj.style.display = 'flex';
      if (trajToolbarRow) trajToolbarRow.style.display = 'flex';

      if (tabBtnSample) tabBtnSample.style.display = 'none';
      if (tabBtnTraj) tabBtnTraj.style.display = 'inline-block';
      this.setActiveTab('trajectory');

      this.trajectoryGroup.visible = true;
      this.resetTrajectoryPlaybackClock();
      if (this.waypointMarkersGroup) {
        this.waypointMarkersGroup.visible = (this.trajectoryFamily === 'custom_waypoints');
      }

      // Ensure scene is selected
      if (!this.currentSceneId && this.scenes.length > 0) {
        this.currentSceneId = this.scenes[0];
      }
      const trajSel = document.getElementById('traj-scene-select');
      if (trajSel) trajSel.value = this.currentSceneId || '';

      const trajCtxEl = document.getElementById('traj-quiet-context');
      if (trajCtxEl) trajCtxEl.textContent = `${this.currentSceneId || 'no scene'} · Trajectory Lab`;

      this.loadTrajectory();
    }
    this.handleResize();
  }

  setActiveTab(tabName) {
    this.activeTab = tabName;
    const tabBtnSample = document.getElementById('tab-btn-sample');
    const tabBtnTraj = document.getElementById('tab-btn-trajectory');
    const tabBtnObject = document.getElementById('tab-btn-object');

    const panelSample = document.getElementById('tab-panel-sample');
    const panelTraj = document.getElementById('tab-panel-trajectory');
    const panelObject = document.getElementById('tab-panel-object');

    [tabBtnSample, tabBtnTraj, tabBtnObject].forEach((btn) => {
      if (btn) {
        btn.classList.remove('active');
        btn.setAttribute('aria-selected', 'false');
      }
    });
    [panelSample, panelTraj, panelObject].forEach((p) => {
      if (p) p.style.display = 'none';
    });

    if (tabName === 'sample' && tabBtnSample && panelSample) {
      tabBtnSample.classList.add('active');
      tabBtnSample.setAttribute('aria-selected', 'true');
      panelSample.style.display = 'flex';
    } else if (tabName === 'trajectory' && tabBtnTraj && panelTraj) {
      tabBtnTraj.classList.add('active');
      tabBtnTraj.setAttribute('aria-selected', 'true');
      panelTraj.style.display = 'flex';
    } else if (tabName === 'object' && tabBtnObject && panelObject) {
      tabBtnObject.classList.add('active');
      tabBtnObject.setAttribute('aria-selected', 'true');
      panelObject.style.display = 'flex';
    }
  }

  // =========================================================================
  // Experiment Drawer
  // =========================================================================
  toggleExperimentDrawer(open) {
    const drawer = document.getElementById('experiment-drawer');
    const backdrop = document.getElementById('drawer-backdrop');
    const btn = document.getElementById('btn-experiment-drawer');

    if (open) {
      drawer.style.display = 'flex';
      backdrop.style.display = 'block';
      drawer.setAttribute('aria-hidden', 'false');
      btn.setAttribute('aria-expanded', 'true');
    } else {
      drawer.style.display = 'none';
      backdrop.style.display = 'none';
      drawer.setAttribute('aria-hidden', 'true');
      btn.setAttribute('aria-expanded', 'false');
    }
  }

  updateExperimentContextButton() {
    const label = document.getElementById('exp-context-label');
    if (!this.currentRunId) {
      label.textContent = 'Experiment · Pure Scene';
      return;
    }

    const r = this.runs.find((x) => x.run_id === this.currentRunId);
    if (r) {
      if (r.group && r.seed !== null && r.seed !== undefined) {
        label.textContent = `Experiment · Group ${r.group} / Seed ${r.seed}`;
      } else if (r.group) {
        label.textContent = `Experiment · Baseline ${r.group}`;
      } else {
        label.textContent = `Experiment · ${r.run_id}`;
      }
    } else {
      label.textContent = `Experiment · ${this.currentRunId}`;
    }
  }

  updateDrawerRunDetails() {
    const r = this.runs.find((x) => x.run_id === this.currentRunId);
    const modelEl = document.getElementById('drawer-model');
    const seedEl = document.getElementById('drawer-seed');
    const groupEl = document.getElementById('drawer-group');
    const statusEl = document.getElementById('drawer-status');

    if (!r) {
      modelEl.textContent = 'None (Pure 3D Scene)';
      seedEl.textContent = '-';
      groupEl.textContent = '-';
      statusEl.textContent = '-';
      return;
    }

    modelEl.textContent = r.model_id || (r.has_manifest ? 'Qwen/Qwen2.5-VL-3B-Instruct' : 'N/A');
    seedEl.textContent = r.seed !== null && r.seed !== undefined ? r.seed : '-';
    groupEl.textContent = r.group ? `Group ${r.group}` : '-';
    statusEl.textContent = r.status || (r.has_progress ? 'running' : 'completed');
  }

  updateDrawerProfileDetails(profileId) {
    const prof = this.profiles.find((p) => p.profile_id === profileId);
    if (!prof) return;

    document.getElementById('spec-mb').textContent = prof.microbatch_size;
    document.getElementById('spec-acc').textContent = prof.gradient_accumulation_steps;
    document.getElementById('spec-eff').textContent = prof.effective_batch_size;
    document.getElementById('spec-mem').textContent = `~${(prof.memory_estimate_mb / 1024).toFixed(1)} GB`;
  }

  // =========================================================================
  // Data Loading
  // =========================================================================
  async loadProfiles() {
    try {
      const res = await fetch('/api/profiles');
      const data = await res.json();
      this.profiles = data.profiles || [];
      const sel = document.getElementById('profile-select');
      sel.innerHTML = '';
      this.profiles.forEach((p) => {
        const opt = document.createElement('option');
        opt.value = p.profile_id;
        opt.textContent = `${p.display_name} (MB${p.microbatch_size}×ACC${p.gradient_accumulation_steps}, ~${(p.memory_estimate_mb / 1024).toFixed(1)}G)`;
        if (p.profile_id === data.default_profile_id) opt.selected = true;
        sel.appendChild(opt);
      });
      if (data.default_profile_id) {
        this.updateDrawerProfileDetails(data.default_profile_id);
      }
    } catch (e) {
      console.error('Failed to load profiles:', e);
    }
  }

  async loadRuns() {
    try {
      const res = await fetch('/api/runs');
      const data = await res.json();
      this.runs = data.runs || [];
      const sel = document.getElementById('run-select');
      sel.innerHTML = '<option value="">-- Pure Scene (No Run) --</option>';

      // Default to completed run with predictions (Group D Seed 42 if present)
      let defaultRunId = null;

      this.runs.forEach((r) => {
        const opt = document.createElement('option');
        opt.value = r.run_id;
        const tag = r.has_predictions ? '[Eval]' : r.has_progress ? '[Train]' : '';
        let label = r.run_id;
        if (r.group && r.seed) {
          label = `Group ${r.group} / Seed ${r.seed} (${r.run_id})`;
        } else if (r.group) {
          label = `Baseline ${r.group} (${r.run_id})`;
        }
        opt.textContent = `${tag} ${label}`;
        sel.appendChild(opt);

        if (!defaultRunId && r.run_id === 'g2.0-e3/group_d/seed_42') {
          defaultRunId = r.run_id;
        } else if (!defaultRunId && r.has_predictions) {
          defaultRunId = r.run_id;
        }
      });

      if (defaultRunId) {
        this.currentRunId = defaultRunId;
        sel.value = defaultRunId;
      }
      this.updateExperimentContextButton();
      this.updateDrawerRunDetails();
    } catch (e) {
      console.error('Failed to load runs:', e);
    }
  }

  async loadAllSamples() {
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      this.allSamples = data.samples || [];

      // Also populate hidden select for compatibility
      const hiddenSel = document.getElementById('sample-select');
      hiddenSel.innerHTML = '';
      this.allSamples.forEach((s) => {
        const opt = document.createElement('option');
        opt.value = s.id || s.sample_id;
        opt.textContent = s.question;
        hiddenSel.appendChild(opt);
      });

      if (this.allSamples.length > 0 && !this.currentSampleId) {
        // Prefer a sample in scene_080 (which has completed eval predictions in Group D)
        const s80 = this.allSamples.find((s) => s.scene_id === 'scene_080');
        const initial = s80 || this.allSamples[0];
        this.selectSample(initial, false);
      }
    } catch (e) {
      console.error('Failed to load samples:', e);
    }
  }

  async loadScenes() {
    try {
      const res = await fetch('/api/scenes');
      const data = await res.json();
      this.scenes = data.scenes || [];
      const sel = document.getElementById('scene-select');
      sel.innerHTML = '';
      this.scenes.forEach((s) => {
        const opt = document.createElement('option');
        opt.value = s;
        opt.textContent = s;
        sel.appendChild(opt);
      });

      const trajSel = document.getElementById('traj-scene-select');
      if (trajSel) {
        trajSel.innerHTML = '';
        this.scenes.forEach((s) => {
          const opt = document.createElement('option');
          opt.value = s;
          opt.textContent = s;
          trajSel.appendChild(opt);
        });
      }

      if (!this.currentSceneId && this.scenes.length > 0) {
        this.currentSceneId = this.scenes[0];
      }
      sel.value = this.currentSceneId || '';
      if (trajSel) trajSel.value = this.currentSceneId || '';
      await this.loadViewSets(this.currentSceneId);
    } catch (e) {
      console.error('Failed to load scenes:', e);
    }
  }

  async loadViewSets(sceneId) {
    const sId = sceneId || this.currentSceneId || 'scene_000';
    try {
      const res = await fetch(`/api/view_sets?scene_id=${sId}`);
      if (!res.ok) return;
      this.viewSetsData = await res.json();

      const vsSel = document.getElementById('viewset-select');
      if (vsSel && this.viewSetsData.view_sets) {
        const currentVs = vsSel.value || this.currentViewSetId || 'historical_8';
        vsSel.innerHTML = '';
        this.viewSetsData.view_sets.forEach((vs) => {
          const opt = document.createElement('option');
          opt.value = vs.id;
          opt.textContent = `${vs.name} (${vs.count})`;
          vsSel.appendChild(opt);
        });
        vsSel.value = currentVs;
        this.currentViewSetId = currentVs;
      }
      this.applyActiveViewSet();
    } catch (e) {
      console.error('Failed to load view sets:', e);
    }
  }

  applyActiveViewSet() {
    if (!this.viewSetsData || !this.viewSetsData.view_sets) return;
    const vs = this.viewSetsData.view_sets.find((s) => s.id === this.currentViewSetId);
    if (!vs) return;

    this.views = vs.view_ids || [];
    const sel = document.getElementById('view-select');
    if (sel) {
      sel.innerHTML = '';
      this.views.forEach((v) => {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        sel.appendChild(opt);
      });

      if (!this.currentViewId || !this.views.includes(this.currentViewId)) {
        this.currentViewId = this.views[0] || 'view_0';
      }
      sel.value = this.currentViewId;
    }

    this.updateViewStepperUI();
    if (this.showCanonicalCams) {
      this.renderCanonicalCameraMarkers();
    }
  }

  updateViewStepperUI() {
    const idx = this.views.indexOf(this.currentViewId);
    const badge = document.getElementById('view-index-badge');
    const prevBtn = document.getElementById('btn-view-prev');
    const nextBtn = document.getElementById('btn-view-next');

    if (badge) {
      const pos = idx >= 0 ? idx + 1 : 1;
      badge.textContent = `${pos} / ${this.views.length}`;
    }
    if (prevBtn) prevBtn.disabled = (idx <= 0);
    if (nextBtn) nextBtn.disabled = (idx >= this.views.length - 1 || idx < 0);
  }

  onViewSetChange(viewSetId) {
    this.currentViewSetId = viewSetId;
    this.applyActiveViewSet();
    this.loadSnapshot();
  }

  stepView(delta) {
    if (!this.views.length) return;
    const currentIdx = this.views.indexOf(this.currentViewId);
    let nextIdx = currentIdx + delta;
    if (nextIdx < 0) nextIdx = 0;
    if (nextIdx >= this.views.length) nextIdx = this.views.length - 1;

    this.currentViewId = this.views[nextIdx];
    const sel = document.getElementById('view-select');
    if (sel) sel.value = this.currentViewId;

    this.updateViewStepperUI();
    if (this.showCanonicalCams) {
      this.renderCanonicalCameraMarkers();
    }
    this.loadSnapshot();
  }

  toggleCanonicalCameras() {
    this.showCanonicalCams = !this.showCanonicalCams;
    const btn = document.getElementById('btn-toggle-canonical-cams');
    if (btn) {
      btn.textContent = this.showCanonicalCams ? 'Cameras: 26 (On)' : 'Cameras: 26 (Off)';
      btn.classList.toggle('active', this.showCanonicalCams);
    }
    this.renderCanonicalCameraMarkers();
  }

  renderCanonicalCameraMarkers() {
    while (this.canonicalCamsGroup.children.length > 0) {
      const c = this.canonicalCamsGroup.children.pop();
      if (c.geometry) c.geometry.dispose();
      if (c.material) c.material.dispose();
    }

    if (!this.showCanonicalCams || !this.viewSetsData || !this.viewSetsData.canonical_poses) {
      return;
    }

    const poses = this.viewSetsData.canonical_poses;
    Object.keys(poses).forEach((viewKey) => {
      const p = poses[viewKey];
      const pos = new THREE.Vector3(...p.position);
      const look = new THREE.Vector3(...p.look_at);
      const isActive = (viewKey === this.currentViewId);

      // Mini-cone for camera direction
      const coneGeom = new THREE.ConeGeometry(0.12, 0.25, 4);
      coneGeom.rotateX(Math.PI / 2);
      const coneMat = new THREE.MeshBasicMaterial({
        color: isActive ? 0xd4c28f : 0x62798a,
        wireframe: true,
      });
      const coneMesh = new THREE.Mesh(coneGeom, coneMat);
      coneMesh.position.copy(pos);
      coneMesh.lookAt(look);
      coneMesh.userData.view_id = viewKey;
      this.canonicalCamsGroup.add(coneMesh);

      // Line connecting to center
      const rayGeom = new THREE.BufferGeometry().setFromPoints([pos, look]);
      const rayMat = new THREE.LineBasicMaterial({
        color: isActive ? 0xd4c28f : 0x282e35,
        transparent: true,
        opacity: isActive ? 0.9 : 0.35,
      });
      const rayLine = new THREE.Line(rayGeom, rayMat);
      rayLine.userData.view_id = viewKey;
      this.canonicalCamsGroup.add(rayLine);
    });
  }

  applyCameraDistance(distance) {
    if (!this.currentSnapshot || !this.currentSnapshot.camera || !this.currentSnapshot.camera.pose) return;
    const pose = this.currentSnapshot.camera.pose;
    const target = new THREE.Vector3(...pose.look_at);
    const oldPos = new THREE.Vector3(...pose.position);
    const dir = new THREE.Vector3().subVectors(oldPos, target).normalize();
    const newPos = new THREE.Vector3().addVectors(target, dir.multiplyScalar(distance));
    const newPose = {
      ...pose,
      position: [newPos.x, newPos.y, newPos.z],
    };
    this.render3DCameraPose(newPose, 0x8da6b8, 4.2);
    const badge = document.getElementById('explore-actual-dist');
    if (badge) badge.textContent = `${distance.toFixed(2)}m`;
  }

  // =========================================================================
  // Custom Waypoint Spline Editor Logic
  // =========================================================================
  initWaypointEditor() {
    this.renderWaypointTable();
    this.renderWaypoint3D();
  }

  renderWaypointTable() {
    const tbody = document.getElementById('waypoint-tbody');
    const countBadge = document.getElementById('waypoint-count-badge');
    const summaryText = document.getElementById('waypoint-summary-text');
    if (!tbody) return;

    tbody.innerHTML = '';
    if (countBadge) countBadge.textContent = `${this.waypoints.length} pts`;
    if (summaryText) {
      summaryText.textContent = `Points: ${this.waypoints.length} · Mode: ${this.waypointInterpMode === 'catmull_rom' ? 'Centripetal Catmull-Rom' : 'Linear'} · Resampled`;
    }

    this.waypoints.forEach((wp, i) => {
      const tr = document.createElement('tr');
      const targetStr = (wp.tx !== undefined && wp.ty !== undefined && wp.tz !== undefined)
        ? `${wp.tx.toFixed(1)}, ${wp.ty.toFixed(1)}, ${wp.tz.toFixed(1)}`
        : '0.0, 0.0, 0.4';

      tr.innerHTML = `
        <td class="mono" style="color: var(--text-muted);">${i + 1}</td>
        <td><input type="number" step="0.1" class="waypoint-coord-input mono" value="${wp.x}" data-idx="${i}" data-field="x"></td>
        <td><input type="number" step="0.1" class="waypoint-coord-input mono" value="${wp.y}" data-idx="${i}" data-field="y"></td>
        <td><input type="number" step="0.1" class="waypoint-coord-input mono" value="${wp.z}" data-idx="${i}" data-field="z"></td>
        <td><input type="text" class="waypoint-target-input mono" value="${targetStr}" data-idx="${i}" data-field="target" placeholder="tx, ty, tz"></td>
        <td style="text-align: right; white-space: nowrap;">
          <button type="button" class="waypoint-row-btn wp-up" data-idx="${i}" title="Move up" ${i === 0 ? 'disabled' : ''}>↑</button>
          <button type="button" class="waypoint-row-btn wp-down" data-idx="${i}" title="Move down" ${i === this.waypoints.length - 1 ? 'disabled' : ''}>↓</button>
          <button type="button" class="waypoint-row-btn waypoint-row-del wp-del" data-idx="${i}" title="Delete" ${this.waypoints.length <= 2 ? 'disabled' : ''}>✕</button>
        </td>
      `;
      tbody.appendChild(tr);
    });

    tbody.querySelectorAll('.waypoint-coord-input').forEach((input) => {
      input.addEventListener('change', (e) => {
        const idx = parseInt(e.target.dataset.idx, 10);
        const field = e.target.dataset.field;
        const val = parseFloat(e.target.value);
        if (!isNaN(val) && this.waypoints[idx]) {
          this.waypoints[idx][field] = val;
          this.renderWaypoint3D();
        }
      });
    });

    tbody.querySelectorAll('.waypoint-target-input').forEach((input) => {
      input.addEventListener('change', (e) => {
        const idx = parseInt(e.target.dataset.idx, 10);
        const parts = e.target.value.split(',').map((s) => parseFloat(s.trim()));
        if (parts.length >= 3 && !parts.some(isNaN) && this.waypoints[idx]) {
          this.waypoints[idx].tx = parts[0];
          this.waypoints[idx].ty = parts[1];
          this.waypoints[idx].tz = parts[2];
        }
      });
    });

    tbody.querySelectorAll('.wp-up').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const idx = parseInt(e.currentTarget.dataset.idx, 10);
        if (idx > 0) {
          const temp = this.waypoints[idx - 1];
          this.waypoints[idx - 1] = this.waypoints[idx];
          this.waypoints[idx] = temp;
          this.renderWaypointTable();
          this.renderWaypoint3D();
        }
      });
    });

    tbody.querySelectorAll('.wp-down').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const idx = parseInt(e.currentTarget.dataset.idx, 10);
        if (idx < this.waypoints.length - 1) {
          const temp = this.waypoints[idx + 1];
          this.waypoints[idx + 1] = this.waypoints[idx];
          this.waypoints[idx] = temp;
          this.renderWaypointTable();
          this.renderWaypoint3D();
        }
      });
    });

    tbody.querySelectorAll('.wp-del').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const idx = parseInt(e.currentTarget.dataset.idx, 10);
        if (this.waypoints.length > 2) {
          this.waypoints.splice(idx, 1);
          this.renderWaypointTable();
          this.renderWaypoint3D();
        }
      });
    });
  }

  renderWaypoint3D() {
    while (this.waypointMarkersGroup.children.length > 0) {
      const c = this.waypointMarkersGroup.children.pop();
      if (c.geometry) c.geometry.dispose();
      if (c.material) c.material.dispose();
    }

    if (this.trajectoryFamily !== 'custom_waypoints') {
      return;
    }

    if (!this.waypoints || this.waypoints.length < 2) return;

    const points = this.waypoints.map((wp) => new THREE.Vector3(wp.x, wp.y, wp.z));

    points.forEach((pt, i) => {
      const geom = new THREE.SphereGeometry(0.12, 16, 12);
      const mat = new THREE.MeshBasicMaterial({
        color: (i === 0) ? 0x789b83 : (i === points.length - 1 ? 0xad7474 : 0xf59e0b),
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.copy(pt);
      this.waypointMarkersGroup.add(mesh);
    });

    let splinePoints = [];
    if (this.waypointInterpMode === 'catmull_rom' && points.length >= 2) {
      const curve = new THREE.CatmullRomCurve3(points, false, 'centripetal');
      splinePoints = curve.getPoints(Math.max(50, points.length * 20));
    } else {
      splinePoints = points;
    }

    const lineGeom = new THREE.BufferGeometry().setFromPoints(splinePoints);
    const lineMat = new THREE.LineBasicMaterial({
      color: 0xf59e0b,
      linewidth: 2,
      transparent: true,
      opacity: 0.85,
    });
    const line = new THREE.Line(lineGeom, lineMat);
    this.waypointMarkersGroup.add(line);
  }

  addWaypoint() {
    const len = this.waypoints.length;
    let newPt;
    if (len >= 2) {
      const p1 = this.waypoints[len - 2];
      const p2 = this.waypoints[len - 1];
      newPt = {
        x: parseFloat((p2.x + (p2.x - p1.x) * 0.5).toFixed(2)),
        y: parseFloat((p2.y + (p2.y - p1.y) * 0.5).toFixed(2)),
        z: parseFloat((p2.z + (p2.z - p1.z) * 0.5).toFixed(2)),
        tx: p2.tx !== undefined ? p2.tx : 0.0,
        ty: p2.ty !== undefined ? p2.ty : 0.0,
        tz: p2.tz !== undefined ? p2.tz : 0.4,
      };
    } else {
      newPt = { x: 3.0, y: -3.0, z: 2.0, tx: 0.0, ty: 0.0, tz: 0.4 };
    }
    this.waypoints.push(newPt);
    this.renderWaypointTable();
    this.renderWaypoint3D();
  }

  loadDefaultWaypoints() {
    this.waypoints = [
      { x: 3.5, y: -3.5, z: 2.2, tx: 0.0, ty: 0.0, tz: 0.4 },
      { x: 0.0, y: -4.5, z: 2.8, tx: 0.0, ty: 0.0, tz: 0.4 },
      { x: -3.5, y: -3.5, z: 2.2, tx: 0.0, ty: 0.0, tz: 0.4 },
      { x: 0.0, y: -2.0, z: 1.5, tx: 0.0, ty: 0.0, tz: 0.4 },
    ];
    this.renderWaypointTable();
    this.renderWaypoint3D();
    this.loadTrajectory();
  }

  clearWaypoints() {
    this.waypoints = [
      { x: 3.0, y: -3.0, z: 2.0, tx: 0.0, ty: 0.0, tz: 0.4 },
      { x: -3.0, y: -3.0, z: 2.0, tx: 0.0, ty: 0.0, tz: 0.4 },
    ];
    this.renderWaypointTable();
    this.renderWaypoint3D();
  }

  applyWaypointPaste() {
    const textarea = document.getElementById('waypoint-paste-textarea');
    const feedback = document.getElementById('waypoint-paste-feedback');
    if (!textarea) return;

    const text = textarea.value;
    const lines = text.split('\n');
    const parsed = [];

    for (let rawLine of lines) {
      const line = rawLine.split('#')[0].trim();
      if (!line) continue;
      const tokens = line.split(/[,\s]+/).map((s) => parseFloat(s)).filter((n) => !isNaN(n));
      if (tokens.length >= 6) {
        parsed.push({
          x: tokens[0],
          y: tokens[1],
          z: tokens[2],
          tx: tokens[3],
          ty: tokens[4],
          tz: tokens[5],
        });
      } else if (tokens.length >= 3) {
        parsed.push({
          x: tokens[0],
          y: tokens[1],
          z: tokens[2],
          tx: 0.0,
          ty: 0.0,
          tz: 0.4,
        });
      } else {
        if (feedback) feedback.textContent = `Error: invalid line "${rawLine.trim()}"`;
        return;
      }
    }

    if (parsed.length < 2) {
      if (feedback) feedback.textContent = `Error: at least 2 waypoints required (got ${parsed.length})`;
      return;
    }

    this.waypoints = parsed;
    if (feedback) {
      feedback.textContent = `Successfully imported ${parsed.length} waypoints!`;
      feedback.style.color = 'var(--success)';
      setTimeout(() => { feedback.textContent = ''; }, 3000);
    }
    this.renderWaypointTable();
    this.renderWaypoint3D();
    this.loadTrajectory();
  }

  exportWaypointsToText() {
    const textarea = document.getElementById('waypoint-paste-textarea');
    if (!textarea) return;
    const lines = this.waypoints.map((wp) => {
      if (wp.tx !== undefined && wp.ty !== undefined && wp.tz !== undefined) {
        return `${wp.x.toFixed(2)}, ${wp.y.toFixed(2)}, ${wp.z.toFixed(2)}, ${wp.tx.toFixed(2)}, ${wp.ty.toFixed(2)}, ${wp.tz.toFixed(2)}`;
      }
      return `${wp.x.toFixed(2)}, ${wp.y.toFixed(2)}, ${wp.z.toFixed(2)}`;
    });
    textarea.value = lines.join('\n');
  }

  // =========================================================================
  // Sample Selection & Dropdown Search
  // =========================================================================
  selectSample(sample, reload = true) {
    if (!sample) return;
    const id = sample.id || sample.sample_id;
    this.currentSampleId = id;
    this.currentSceneId = sample.scene_id;
    this.currentViewId = sample.view_id;

    // Update search input text
    const searchInput = document.getElementById('sample-search-input');
    searchInput.value = `[${sample.family}] ${sample.scene_id} · ${sample.view_id}: ${sample.question}`;

    // Update quiet context badge
    const quietContext = document.getElementById('sample-quiet-context');
    quietContext.textContent = `${sample.scene_id} · ${sample.view_id} · ${sample.family}`;

    // Keep hidden select in sync
    const hiddenSel = document.getElementById('sample-select');
    hiddenSel.value = id;

    // Synchronize scene and view dropdowns (in case user switches to Explore mode)
    const sceneSel = document.getElementById('scene-select');
    if (sceneSel && this.scenes.includes(sample.scene_id)) {
      sceneSel.value = sample.scene_id;
    }
    const viewSel = document.getElementById('view-select');
    if (viewSel) {
      viewSel.value = sample.view_id;
    }

    this.closeSampleDropdown();

    if (reload) {
      this.loadSnapshot();
    }
  }

  selectSampleById(sampleId) {
    const s = this.allSamples.find((item) => (item.id || item.sample_id) === sampleId);
    if (s) {
      this.selectSample(s, true);
    }
  }

  stepSample(delta) {
    if (this.allSamples.length === 0) return;
    const currentIdx = this.allSamples.findIndex((s) => (s.id || s.sample_id) === this.currentSampleId);
    let nextIdx = (currentIdx + delta) % this.allSamples.length;
    if (nextIdx < 0) nextIdx += this.allSamples.length;
    this.selectSample(this.allSamples[nextIdx], true);
  }

  filterSampleDropdown(query) {
    const q = (query || '').trim().toLowerCase();
    const dropdown = document.getElementById('sample-dropdown');

    if (!q) {
      this.filteredSamples = this.allSamples.slice(0, 50);
    } else {
      this.filteredSamples = this.allSamples
        .filter((s) => {
          const id = (s.id || s.sample_id || '').toLowerCase();
          const scene = (s.scene_id || '').toLowerCase();
          const view = (s.view_id || '').toLowerCase();
          const fam = (s.family || '').toLowerCase();
          const ques = (s.question || '').toLowerCase();
          const ans = (s.answer || '').toLowerCase();
          return id.includes(q) || scene.includes(q) || view.includes(q) || fam.includes(q) || ques.includes(q) || ans.includes(q);
        })
        .slice(0, 50);
    }

    if (this.filteredSamples.length === 0) {
      dropdown.innerHTML = '<div style="padding:10px 12px;color:var(--text-muted);">No matching samples found</div>';
      dropdown.style.display = 'block';
      this.activeSearchIndex = -1;
      return;
    }

    dropdown.innerHTML = this.filteredSamples
      .map((s, idx) => {
        const id = s.id || s.sample_id;
        const isSelected = id === this.currentSampleId;
        const cls = isSelected ? 'sample-opt selected' : 'sample-opt';
        return `
          <div class="${cls}" data-id="${escapeHtml(id)}" data-idx="${idx}">
            <div class="sample-opt-header">
              <span class="sample-opt-badge">[${escapeHtml(s.family)}]</span>
              <span class="sample-opt-loc">${escapeHtml(s.scene_id)} &middot; ${escapeHtml(s.view_id)}</span>
              <span class="sample-opt-ans">ans: ${escapeHtml(s.answer)}</span>
            </div>
            <div class="sample-opt-q">${escapeHtml(s.question)}</div>
            <div class="sample-opt-id">${escapeHtml(id)}</div>
          </div>
        `;
      })
      .join('');

    dropdown.style.display = 'block';
    this.activeSearchIndex = -1;

    // Attach click listeners
    dropdown.querySelectorAll('.sample-opt').forEach((el) => {
      el.addEventListener('click', () => {
        const sampleId = el.getAttribute('data-id');
        this.selectSampleById(sampleId);
      });
    });
  }

  highlightDropdownItem(items) {
    items.forEach((it, idx) => {
      if (idx === this.activeSearchIndex) {
        it.classList.add('selected');
        it.scrollIntoView({ block: 'nearest' });
      } else {
        it.classList.remove('selected');
      }
    });
  }

  closeSampleDropdown() {
    const dropdown = document.getElementById('sample-dropdown');
    dropdown.style.display = 'none';
    this.activeSearchIndex = -1;
  }

  // =========================================================================
  // Snapshot Loading & Rendering
  // =========================================================================
  async loadSnapshot() {
    if (!this.currentSceneId || !this.currentViewId) return;

    let url = `/api/snapshot?scene_id=${this.currentSceneId}&view_id=${this.currentViewId}`;
    if (this.mode === 'sample' && this.currentSampleId) {
      url += `&sample_id=${encodeURIComponent(this.currentSampleId)}`;
    } else {
      url += '&sample_id=none';
    }
    if (this.currentRunId) {
      url += `&run_id=${encodeURIComponent(this.currentRunId)}`;
    }

    if (this.mode === 'explore') {
      if (this.currentExploreDistanceMode === 'manual') {
        url += `&distance=${this.currentManualDistance}`;
      } else if (this.currentExploreDistanceMode === 'adaptive_scale') {
        url += `&distance_mode=adaptive_scale&distance_scale=${this.currentDistanceScale}`;
      } else if (this.currentExploreDistanceMode === 'adaptive') {
        url += `&distance_mode=adaptive`;
      }
    }

    try {
      const res = await fetch(url);
      if (!res.ok) {
        const err = await res.json();
        console.error('Snapshot error:', err);
        return;
      }
      this.currentSnapshot = await res.json();
      this.renderSnapshot(this.currentSnapshot);
    } catch (e) {
      console.error('Failed to load snapshot:', e);
    }
  }

  renderSnapshot(snap) {
    // 1. Observation Image & Metadata
    const obsImg = document.getElementById('obs-image');
    const proj = snap.camera.projection;
    const isPreview = (proj.status === 'preview_generated' || !snap.observation.image_url);

    let unrenderedNotice = document.getElementById('unrendered-preview-notice');
    if (!unrenderedNotice && obsImg.parentElement) {
      unrenderedNotice = document.createElement('div');
      unrenderedNotice.id = 'unrendered-preview-notice';
      unrenderedNotice.className = 'unrendered-preview-badge';
      unrenderedNotice.innerHTML = '<span>⚠ Unrendered Canonical View (Geometry Preview Only &middot; Not Fabricated Training RGB)</span>';
      obsImg.parentElement.insertBefore(unrenderedNotice, obsImg);
    }

    if (isPreview) {
      if (unrenderedNotice) unrenderedNotice.style.display = 'inline-flex';
      obsImg.style.display = 'none';
    } else {
      if (unrenderedNotice) unrenderedNotice.style.display = 'none';
      obsImg.style.display = 'block';
      obsImg.src = snap.observation.image_url || '';
    }

    // Actual distance badge
    const distBadge = document.getElementById('explore-actual-dist');
    if (distBadge && snap.camera && snap.camera.pose) {
      const p = snap.camera.pose.position;
      const l = snap.camera.pose.look_at;
      const dist = Math.sqrt(
        (p[0] - l[0]) ** 2 +
        (p[1] - l[1]) ** 2 +
        (p[2] - l[2]) ** 2
      );
      distBadge.textContent = `${dist.toFixed(2)}m`;
      if (this.currentExploreDistanceMode === 'manual') {
        const distInput = document.getElementById('explore-dist-input');
        const distSlider = document.getElementById('explore-dist-slider');
        if (distInput && parseFloat(distInput.value) !== this.currentManualDistance) {
          distInput.value = this.currentManualDistance.toFixed(1);
        }
        if (distSlider && parseFloat(distSlider.value) !== this.currentManualDistance) {
          distSlider.value = this.currentManualDistance;
        }
      }
    }

    const metaSceneView = document.getElementById('meta-scene-view');
    metaSceneView.textContent = `${snap.identity.scene_id} · ${snap.identity.view_id}`;

    const resEl = document.getElementById('meta-resolution');
    if (resEl) {
      resEl.textContent = snap.observation.width
        ? `${snap.observation.width} × ${snap.observation.height} px`
        : '512 × 512 px';
    }

    const projStatusEl = document.getElementById('meta-proj-status');
    if (projStatusEl) {
      projStatusEl.textContent = proj.status === 'reconstructed_legacy'
        ? `Reconstructed (${proj.horizontal_fov_deg ? proj.horizontal_fov_deg.toFixed(1) + '°' : 'legacy'})`
        : proj.status;
    }

    const provEl = document.getElementById('meta-provenance');
    if (provEl) {
      provEl.textContent = proj.provenance || '';
    }

    // 2. Mode-Dependent Sections
    if (this.mode === 'sample') {
      // Question
      document.getElementById('qa-question').textContent = snap.model_input.question || 'No question available.';

      // Prediction
      document.getElementById('qa-target').textContent = snap.supervision.ground_truth || 'None';

      const pred = snap.prediction;
      const predVal = pred.parsed_prediction || pred.raw_prediction || 'None';
      document.getElementById('qa-prediction').textContent = predVal;

      const outcomeEl = document.getElementById('qa-outcome');
      const errBox = document.getElementById('qa-alignment-error');
      const rawRow = document.getElementById('qa-raw-row');
      const rawValEl = document.getElementById('qa-raw-val');
      errBox.style.display = 'none';

      if (pred.status === 'available') {
        if (pred.is_correct) {
          outcomeEl.className = 'status-indicator status-correct';
          outcomeEl.textContent = '✓ Correct';
        } else {
          outcomeEl.className = 'status-indicator status-error';
          outcomeEl.textContent = '✕ Incorrect';
        }
      } else if (pred.status === 'alignment_error') {
        outcomeEl.className = 'status-indicator status-align-error';
        outcomeEl.textContent = 'Metadata Conflict';
        errBox.style.display = 'block';
        errBox.textContent = `Metadata Conflict: ${pred.error_message}`;
      } else {
        outcomeEl.className = 'status-indicator status-unavailable';
        outcomeEl.textContent = this.currentRunId ? 'Unavailable (No eval)' : 'Unavailable (Pure scene)';
      }

      // Raw output row if different from parsed
      if (pred.raw_prediction && pred.parsed_prediction && pred.raw_prediction !== pred.parsed_prediction) {
        rawRow.style.display = 'flex';
        rawValEl.textContent = `"${pred.raw_prediction}"`;
      } else {
        rawRow.style.display = 'none';
      }

      // Relevant Spatial Truth
      this.renderSpatialTruthList(snap);
    } else {
      // Explore Scene mode
      const exploreContext = document.getElementById('explore-quiet-context');
      if (exploreContext) {
        exploreContext.textContent = `${snap.environment_truth.objects.length} objects · ${snap.identity.view_id} viewpoint`;
      }
    }

    // Challenge Summary
    if (snap.environment_truth && snap.environment_truth.challenge_metadata) {
      this.renderChallengeSummary(snap.environment_truth.challenge_metadata);
    }

    // 3. 3D Scene Primitives
    this.render3DObjects(snap);

    // 4. 3D Camera & Frustum
    this.render3DCamera(snap);

    // 5. Update Object Selection
    if (this.selectedObjectIndex !== null) {
      this.selectObject(this.selectedObjectIndex, false);
    } else {
      const body = document.getElementById('object-inspector-body');
      body.innerHTML = '<div class="inspector-empty-msg">Select an object in the 3D scene to inspect its geometric state.</div>';
    }

    // 6. Lock/unlock profile selector if run locked
    const profSelect = document.getElementById('profile-select');
    if (snap.identity.run_id && snap.runtime && snap.runtime.training_profile) {
      profSelect.disabled = true;
      profSelect.title = 'Profile locked to inspected run manifest';
    } else {
      profSelect.disabled = false;
      profSelect.title = '';
    }
  }

  renderSpatialTruthList(snap) {
    const listEl = document.getElementById('truth-list');
    const noteEl = document.getElementById('truth-perspective-note');
    const pairTruth = snap.environment_truth.relevant_pair_truth;

    if (!pairTruth) {
      listEl.innerHTML = '<div style="padding:6px 0;color:var(--text-muted);">No pairwise spatial relations for this sample.</div>';
      noteEl.textContent = '';
      return;
    }

    const family = (snap.supervision && snap.supervision.family) ? snap.supervision.family.toLowerCase() : '';

    const rows = [
      { dim: 'Horizontal', token: pairTruth.left_right, key: 'horizontal' },
      { dim: 'Vertical', token: pairTruth.above_below, key: 'vertical' },
      { dim: 'Depth', token: pairTruth.front_behind, key: 'depth' },
      { dim: 'Distance', token: pairTruth.near_far, key: 'distance' },
      { dim: 'Metric Distance', token: `${pairTruth.metric_distance.toFixed(3)} m`, key: 'metric' },
    ];

    listEl.innerHTML = rows
      .map((r) => {
        const isRel = (family === r.key);
        const cls = isRel ? 'truth-item truth-rel is-family-match' : 'truth-item truth-rel';
        return `
          <div class="${cls}">
            <span class="truth-dim">${r.dim}</span>
            <span class="truth-val">${r.token}</span>
          </div>
        `;
      })
      .join('');

    const orderText = pairTruth.presentation_order === 1
      ? `Inverted (Object ${pairTruth.object_index_first} relative to ${pairTruth.object_index_second})`
      : `Canonical (Object ${pairTruth.object_index_first} relative to ${pairTruth.object_index_second})`;
    noteEl.textContent = `${orderText} · Adapted to question perspective`;
  }

  renderObjectsList(objects) {
    while (this.objectsGroup.children.length > 0) {
      const child = this.objectsGroup.children.pop();
      child.geometry.dispose();
      child.material.dispose();
    }

    objects.forEach((obj) => {
      const color = COLOR_MAP[obj.color.toLowerCase()] || 0x888888;
      const mat = new THREE.MeshStandardMaterial({
        color: color,
        roughness: 0.35,
        metalness: 0.1,
      });

      let geom;
      const s = obj.size;
      const shape = (obj.shape || '').toLowerCase();
      if (shape === 'cube') {
        geom = new THREE.BoxGeometry(s, s, s);
      } else if (shape === 'sphere') {
        geom = new THREE.SphereGeometry(s / 2, 32, 24);
      } else if (shape === 'cylinder') {
        geom = new THREE.CylinderGeometry(s / 2, s / 2, s, 32);
        geom.rotateX(Math.PI / 2); // Align depth with Z axis
      } else if (shape === 'cone') {
        geom = new THREE.ConeGeometry(s / 2, s, 32);
        geom.rotateX(Math.PI / 2); // Align height with Z axis (apex +Z)
      } else if (shape === 'torus') {
        geom = new THREE.TorusGeometry(s * 0.32, s * 0.14, 16, 32);
      } else if (shape === 'octahedron') {
        geom = new THREE.OctahedronGeometry(s / 2);
      } else if (shape === 'pyramid') {
        geom = new THREE.ConeGeometry(s / Math.SQRT2, s, 4);
        geom.rotateX(Math.PI / 2);
        geom.rotateZ(Math.PI / 4);
      } else if (shape === 'tetrahedron') {
        geom = new THREE.TetrahedronGeometry(s / 2);
      } else if (shape === 'prism') {
        geom = new THREE.CylinderGeometry(s / 2, s / 2, s, 3);
        geom.rotateX(Math.PI / 2);
      } else if (shape === 'capsule') {
        if (THREE.CapsuleGeometry) {
          geom = new THREE.CapsuleGeometry(s * 0.25, s * 0.5, 8, 16);
          geom.rotateX(Math.PI / 2);
        } else {
          geom = new THREE.CylinderGeometry(s / 2, s / 2, s, 32);
          geom.rotateX(Math.PI / 2);
        }
      } else {
        geom = new THREE.BoxGeometry(s, s, s);
      }

      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(obj.location[0], obj.location[1], obj.location[2]);
      if (obj.rotation && Array.isArray(obj.rotation) && obj.rotation.length === 3) {
        mesh.rotation.set(
          (obj.rotation[0] * Math.PI) / 180,
          (obj.rotation[1] * Math.PI) / 180,
          (obj.rotation[2] * Math.PI) / 180
        );
      }

      mesh.userData = {
        object_index: obj.object_index,
        name: obj.name,
        shape: obj.shape,
        color: obj.color,
        size: obj.size,
        location: obj.location,
        rotation: obj.rotation,
        role: obj.role,
        support_parent: obj.support_parent,
        placement_mode: obj.placement_mode,
        compound_id: obj.compound_id,
        compound_part: obj.compound_part,
      };

      this.objectsGroup.add(mesh);
    });
  }

  updateSupportLines(objects, selectedIndex = null) {
    if (!this.supportLineGroup) return;
    while (this.supportLineGroup.children.length > 0) {
      const child = this.supportLineGroup.children.pop();
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    }

    if (!this.showSupportLines || !objects || objects.length === 0) return;

    const objMap = new Map();
    objects.forEach((o) => objMap.set(o.name, o));

    const selectedObj = selectedIndex !== null ? objects.find((o) => o.object_index === selectedIndex) : null;
    const selectedName = selectedObj ? selectedObj.name : null;

    objects.forEach((childObj) => {
      if (!childObj.support_parent) return;
      const parentObj = objMap.get(childObj.support_parent);
      if (!parentObj) return;

      const isRelated = selectedName && (childObj.name === selectedName || parentObj.name === selectedName);

      const p1 = new THREE.Vector3(parentObj.location[0], parentObj.location[1], parentObj.location[2]);
      const p2 = new THREE.Vector3(childObj.location[0], childObj.location[1], childObj.location[2]);

      const geom = new THREE.BufferGeometry().setFromPoints([p1, p2]);
      const mat = new THREE.LineDashedMaterial({
        color: isRelated ? 0xd4c28f : 0x506577,
        dashSize: 0.08,
        gapSize: 0.04,
        linewidth: 1,
        transparent: true,
        opacity: isRelated ? 0.95 : 0.45,
      });

      const line = new THREE.Line(geom, mat);
      line.computeLineDistances();
      this.supportLineGroup.add(line);
    });
  }

  render3DObjects(snap) {
    this.renderObjectsList(snap.environment_truth.objects);
    this.updateHighlights(snap);
    this.updateSupportLines(snap.environment_truth.objects, this.selectedObjectIndex);
  }

  updateHighlights(snap) {
    while (this.highlightGroup.children.length > 0) {
      const h = this.highlightGroup.children.pop();
      if (h.geometry) h.geometry.dispose();
      if (h.material) h.material.dispose();
    }

    const pair = (this.mode === 'sample') ? snap.environment_truth.relevant_pair_truth : null;
    const targetIdx = pair ? pair.object_index_first : null;
    const refIdx = pair ? pair.object_index_second : null;

    this.objectsGroup.children.forEach((mesh) => {
      const idx = mesh.userData.object_index;
      if (idx === targetIdx || idx === refIdx || idx === this.selectedObjectIndex) {
        const box = new THREE.BoxHelper(mesh);
        if (idx === this.selectedObjectIndex) {
          box.material.color.setHex(0xd4c28f); // Muted warm neutral for user selected
        } else if (idx === targetIdx) {
          box.material.color.setHex(0x8da6b8); // Steel blue accent for question target
        } else if (idx === refIdx) {
          box.material.color.setHex(0xaa9368); // Muted amber for question reference
        }
        this.highlightGroup.add(box);
      }
    });
  }

  render3DCameraPose(pose, colorHex = 0x8da6b8, customFar = null) {
    while (this.cameraMarkerGroup.children.length > 0) {
      const c = this.cameraMarkerGroup.children.pop();
      if (c.geometry) c.geometry.dispose();
      if (c.material) c.material.dispose();
    }

    const C = new THREE.Vector3(...pose.position);
    const L = new THREE.Vector3(...pose.look_at);
    const F = new THREE.Vector3().subVectors(L, C).normalize();
    const UpRaw = new THREE.Vector3(...pose.up);
    const R = new THREE.Vector3().crossVectors(F, UpRaw).normalize();
    const U = new THREE.Vector3().crossVectors(R, F).normalize();

    // Camera body cone
    const camGeom = new THREE.ConeGeometry(0.2, 0.4, 4);
    camGeom.rotateX(Math.PI / 2);
    const camMat = new THREE.MeshBasicMaterial({ color: colorHex, wireframe: true });
    const camMesh = new THREE.Mesh(camGeom, camMat);
    camMesh.position.copy(C);
    camMesh.lookAt(L);
    this.cameraMarkerGroup.add(camMesh);

    // Optical viewing ray
    const rayGeom = new THREE.BufferGeometry().setFromPoints([C, L]);
    const rayMat = new THREE.LineDashedMaterial({ color: colorHex, dashSize: 0.2, gapSize: 0.1 });
    const rayLine = new THREE.Line(rayGeom, rayMat);
    rayLine.computeLineDistances();
    this.cameraMarkerGroup.add(rayLine);

    // Frustum Wireframe
    const fovH = pose.fov_deg || 60.0;
    const fovV = pose.fov_deg || 60.0;
    const tanH = Math.tan(THREE.MathUtils.degToRad(fovH / 2));
    const tanV = Math.tan(THREE.MathUtils.degToRad(fovV / 2));

    const dNear = 0.25;
    const dFar = customFar !== null ? customFar : Math.min(3.5, C.distanceTo(L));

    function corner(dist, sx, sy) {
      return C.clone()
        .addScaledVector(F, dist)
        .addScaledVector(R, sx * dist * tanH)
        .addScaledVector(U, sy * dist * tanV);
    }

    const nTL = corner(dNear, -1, 1);
    const nTR = corner(dNear, 1, 1);
    const nBR = corner(dNear, 1, -1);
    const nBL = corner(dNear, -1, -1);

    const fTL = corner(dFar, -1, 1);
    const fTR = corner(dFar, 1, 1);
    const fBR = corner(dFar, 1, -1);
    const fBL = corner(dFar, -1, -1);

    const points = [
      nTL, nTR, nTR, nBR, nBR, nBL, nBL, nTL,
      fTL, fTR, fTR, fBR, fBR, fBL, fBL, fTL,
      nTL, fTL, nTR, fTR, nBR, fBR, nBL, fBL,
      C, nTL, C, nTR, C, nBR, C, nBL,
    ];

    const fGeom = new THREE.BufferGeometry().setFromPoints(points);
    const fMat = new THREE.LineBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity: 0.45,
    });
    const frustumLines = new THREE.LineSegments(fGeom, fMat);
    this.cameraMarkerGroup.add(frustumLines);
  }

  render3DCamera(snap) {
    const proj = snap.camera.projection;
    const dFar = (proj && proj.is_frustum_available) ? 4.2 : null;
    this.render3DCameraPose(snap.camera.pose, 0x8da6b8, dFar);
  }

  // =========================================================================
  // Scene Challenge Summary & Profile (G2.1-C Restrained UI)
  // =========================================================================
  renderChallengeSummary(meta) {
    if (!meta) return;
    const badge = document.getElementById('challenge-tier-badge');
    if (badge) {
      const tier = meta.difficulty_tier || 'S0';
      const label = tier === 'S0' ? 'Baseline' : tier === 'S1' ? 'Mild' : tier === 'S2' ? 'Strong' : 'Dense';
      badge.textContent = `${tier} · ${label}`;
      badge.className = `badge-challenge-tier mono tier-${tier.toLowerCase()}`;
    }

    const setVal = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };

    setVal('ch-metric-objects', `${meta.object_count} objects`);
    if (meta.depth_layers) {
      setVal('ch-metric-layers', `${meta.depth_layer_count} layers (FG:${meta.depth_layers.foreground} MG:${meta.depth_layers.midground} BG:${meta.depth_layers.background})`);
    }
    setVal('ch-metric-clutter', `${meta.clutter_score}`);
    if (meta.packing_summary) {
      setVal('ch-metric-density', `${(meta.packing_summary.packing_density * 100).toFixed(1)}% (clear: ${meta.packing_summary.min_surface_clearance_m}m)`);
    }
    setVal('ch-metric-elevated', `${meta.elevated_object_count || 0}`);
    setVal('ch-metric-stacked', `${meta.stacked_object_count || 0}`);
    setVal('ch-metric-support', `${meta.support_relation_count || 0} relations`);
    setVal('ch-metric-orientation', `${meta.orientation_diverse_object_count || 0}`);
    setVal('ch-metric-compound', `${meta.compound_object_count || 0}`);
    setVal('ch-metric-overlap', `${meta.overlap_candidate_count} candidates`);
    setVal('ch-metric-distractors', `${meta.distractor_pair_count} pairs`);
    setVal('ch-metric-confounds', `${meta.confound_pair_count} pairs`);

    // Support relations list
    const supBlock = document.getElementById('ch-detail-support');
    const supList = document.getElementById('ch-support-list');
    if (supBlock && supList) {
      if (meta.support_relations && meta.support_relations.length > 0) {
        supBlock.style.display = 'flex';
        supList.innerHTML = meta.support_relations.slice(0, 6).map((r) => {
          return `<li><span class="mono">${escapeHtml(r.child_name)}</span> on <span class="mono">${escapeHtml(r.parent_name)}</span> (${escapeHtml(r.relation_type)}, Δz ${r.contact_z_m}m)</li>`;
        }).join('');
      } else {
        supBlock.style.display = 'none';
      }
    }

    // Compound assemblies list
    const cmpBlock = document.getElementById('ch-detail-compound');
    const cmpList = document.getElementById('ch-compound-list');
    if (cmpBlock && cmpList) {
      if (meta.compound_assemblies && meta.compound_assemblies.length > 0) {
        cmpBlock.style.display = 'flex';
        cmpList.innerHTML = meta.compound_assemblies.slice(0, 4).map((a) => {
          const parts = (a.parts || []).map((p) => `${escapeHtml(p.name)} [${escapeHtml(p.part)}]`).join(', ');
          return `<li><span class="mono">${escapeHtml(a.compound_id)}</span> (${escapeHtml(a.assembly_type)}): ${parts}</li>`;
        }).join('');
      } else {
        cmpBlock.style.display = 'none';
      }
    }

    // Placement modes summary
    const modesBlock = document.getElementById('ch-detail-modes');
    const modesSum = document.getElementById('ch-modes-summary');
    if (modesBlock && modesSum) {
      if (meta.placement_modes && Object.keys(meta.placement_modes).length > 0) {
        modesBlock.style.display = 'flex';
        modesSum.textContent = Object.entries(meta.placement_modes)
          .map(([m, c]) => `${m}: ${c}`)
          .join(' · ');
      } else {
        modesBlock.style.display = 'none';
      }
    }

    // Distractor list
    const distBlock = document.getElementById('ch-detail-distractors');
    const distList = document.getElementById('ch-distractor-list');
    if (distBlock && distList) {
      if (meta.distractor_pairs && meta.distractor_pairs.length > 0) {
        distBlock.style.display = 'flex';
        distList.innerHTML = meta.distractor_pairs.slice(0, 5).map((p) => {
          const typeStr = p.distractor_type === 'similar_color_same_shape' ? 'similar color' :
                          p.distractor_type === 'same_color_different_shape' ? 'same color, diff shape' : 'same shape, diff color';
          return `<li><span class="mono">${escapeHtml(p.name_a)}</span> vs <span class="mono">${escapeHtml(p.name_b)}</span> (${typeStr})</li>`;
        }).join('');
      } else {
        distBlock.style.display = 'none';
      }
    }

    // Overlap list
    const ovBlock = document.getElementById('ch-detail-overlap');
    const ovList = document.getElementById('ch-overlap-list');
    if (ovBlock && ovList) {
      if (meta.overlap_candidates && meta.overlap_candidates.length > 0) {
        ovBlock.style.display = 'flex';
        ovList.innerHTML = meta.overlap_candidates.slice(0, 4).map((c) => {
          return `<li>[${escapeHtml(c.view_id)}] <span class="mono">${escapeHtml(c.near_object)}</span> in front of <span class="mono">${escapeHtml(c.far_object)}</span> (sep ${c.angular_separation_deg}°, Δz ${c.depth_difference_m}m)</li>`;
        }).join('');
      } else {
        ovBlock.style.display = 'none';
      }
    }

    // Confound list
    const confBlock = document.getElementById('ch-detail-confounds');
    const confList = document.getElementById('ch-confound-list');
    if (confBlock && confList) {
      if (meta.confound_pairs && meta.confound_pairs.length > 0) {
        confBlock.style.display = 'flex';
        confList.innerHTML = meta.confound_pairs.slice(0, 3).map((cp) => {
          return `<li><span class="mono">${escapeHtml(cp.small_object)}</span> (s=${cp.small_physical_size}m, d=${cp.near_distance_m}m) vs <span class="mono">${escapeHtml(cp.large_object)}</span> (s=${cp.large_physical_size}m, d=${cp.far_distance_m}m, ratio ${cp.projected_size_ratio})</li>`;
        }).join('');
      } else {
        confBlock.style.display = 'none';
      }
    }
  }

  updateTrajectoryChallengeUI(meta) {
    if (!meta) return;
    const badge = document.getElementById('traj-challenge-tier-badge');
    if (badge) {
      const tier = meta.difficulty_tier || 'S0';
      const label = tier === 'S0' ? 'Baseline' : tier === 'S1' ? 'Mild' : tier === 'S2' ? 'Strong' : 'Dense';
      badge.textContent = `${tier} · ${label}`;
      badge.className = `badge-challenge-tier mono tier-${tier.toLowerCase()}`;
    }

    const setVal = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };

    setVal('traj-ch-metric-objects', `${meta.object_count}`);
    if (meta.depth_layers) {
      setVal('traj-ch-metric-layers', `${meta.depth_layer_count} (FG:${meta.depth_layers.foreground} MG:${meta.depth_layers.midground} BG:${meta.depth_layers.background})`);
    }
    setVal('traj-ch-metric-clutter', `${meta.clutter_score}`);
    setVal('traj-ch-metric-elevated', `${meta.elevated_object_count || 0}`);
    setVal('traj-ch-metric-stacked', `${meta.stacked_object_count || 0}`);
    setVal('traj-ch-metric-support', `${meta.support_relation_count || 0}`);
    setVal('traj-ch-metric-orientation', `${meta.orientation_diverse_object_count || 0}`);
    setVal('traj-ch-metric-compound', `${meta.compound_object_count || 0}`);
    setVal('traj-ch-metric-overlap', `${meta.overlap_candidate_count}`);
    setVal('traj-ch-metric-distractors', `${meta.distractor_pair_count}`);
    setVal('traj-ch-metric-confounds', `${meta.confound_pair_count}`);

    const note = document.getElementById('traj-ch-summary-note');
    if (note && meta.packing_summary) {
      note.textContent = `Clutter score ${meta.clutter_score} · Density ${(meta.packing_summary.packing_density * 100).toFixed(1)}% · Elevated: ${meta.elevated_object_count || 0} · Stacked: ${meta.stacked_object_count || 0} · Support: ${meta.support_relation_count || 0} · Min clearance: ${meta.packing_summary.min_surface_clearance_m}m`;
    }
  }

  // =========================================================================
  // Trajectory Lab & Dynamic Observation Prototype
  // =========================================================================
  async loadTrajectory() {
    const sceneId = document.getElementById('traj-scene-select')?.value || this.currentSceneId || 'scene_000';
    const family = document.getElementById('traj-family-select')?.value || this.trajectoryFamily || 'survey_orbit';
    const framingMode = document.getElementById('traj-framing-select')?.value || 'balanced';
    const frameCount = parseInt(document.getElementById('traj-frames-select')?.value || '60', 10);
    const seed = parseInt(document.getElementById('traj-seed-input')?.value || '0', 10);
    const speed = parseFloat(document.getElementById('traj-speed-input')?.value || '1.0');
    const manualDist = parseFloat(document.getElementById('traj-dist-input')?.value || '3.5');
    const distScale = parseFloat(document.getElementById('traj-scale-select')?.value || '1.0');

    this.trajectoryFamily = family;

    // Advanced overrides
    const azCycles = parseFloat(document.getElementById('traj-cfg-az-cycles')?.value || '1.0');
    const elevBase = parseFloat(document.getElementById('traj-cfg-elev-base')?.value || '25.0');
    const elevAmp = parseFloat(document.getElementById('traj-cfg-elev-amp')?.value || '15.0');
    const elevCycles = parseFloat(document.getElementById('traj-cfg-elev-cycles')?.value || '2.5');
    const radAmp = parseFloat(document.getElementById('traj-cfg-rad-amp')?.value || '0.15');
    const radCycles = parseFloat(document.getElementById('traj-cfg-rad-cycles')?.value || '1.5');
    const biasFrac = parseFloat(document.getElementById('traj-cfg-bias-frac')?.value || '0.10');

    try {
      // 1. Ensure scene objects are loaded into 3D viewport
      const sceneRes = await fetch(`/api/scene/${sceneId}`);
      if (sceneRes.ok) {
        const sceneData = await sceneRes.json();
        if (sceneData.objects) {
          this.renderObjectsList(sceneData.objects);
        }
        if (sceneData.challenge_metadata) {
          this.updateTrajectoryChallengeUI(sceneData.challenge_metadata);
        }
      }

      // 2. Fetch deterministic trajectory
      let res;
      if (family === 'custom_waypoints') {
        res = await fetch('/api/trajectory', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            scene_id: sceneId,
            trajectory_family: family,
            distance_mode: framingMode,
            manual_distance: manualDist,
            distance_scale: distScale,
            speed_m_s: speed,
            frame_count: frameCount,
            seed: seed,
            interpolation_mode: this.waypointInterpMode,
            waypoints_json: JSON.stringify(this.waypoints),
          }),
        });
      } else {
        const queryParams = new URLSearchParams({
          scene_id: sceneId,
          trajectory_family: family,
          distance_mode: framingMode,
          manual_distance: manualDist,
          distance_scale: distScale,
          speed_m_s: speed,
          framing_mode: framingMode,
          frame_count: frameCount,
          seed: seed,
          azimuth_cycles: azCycles,
          elevation_base_deg: elevBase,
          elevation_amplitude_deg: elevAmp,
          elevation_cycles: elevCycles,
          radius_amplitude_frac: radAmp,
          radius_cycles: radCycles,
          target_bias_fraction: biasFrac,
        });
        res = await fetch(`/api/trajectory?${queryParams.toString()}`);
      }

      if (!res.ok) {
        const err = await res.json();
        console.error('Trajectory error:', err);
        return;
      }
      this.currentTrajectory = await res.json();
      this.currentSceneId = sceneId;

      // Update quiet context badge
      const trajCtxEl = document.getElementById('traj-quiet-context');
      if (trajCtxEl) trajCtxEl.textContent = `${sceneId} · ${family} · Trajectory Lab`;

      // Update timeline slider bounds
      const slider = document.getElementById('traj-timeline-slider');
      if (slider) {
        slider.max = Math.max(0, this.currentTrajectory.frames.length - 1);
        slider.value = 0;
      }

      // Render 3D trajectory path
      this.renderTrajectoryPath(this.currentTrajectory);

      // Render waypoint 3D markers if in custom_waypoints
      this.renderWaypoint3D();

      // Populate framing UI card & table
      this.updateFramingUI(this.currentTrajectory);

      // Populate diagnostics UI card
      this.updateDiagnosticsUI(this.currentTrajectory.diagnostics, this.currentTrajectory);

      // Reset playback state and seek to frame 0
      this.pauseTrajectoryPlayback();
      this.resetTrajectoryPlaybackClock();
      this.seekTrajectoryFrame(0, true);
    } catch (e) {
      console.error('Failed to load trajectory:', e);
    }
  }

  renderTrajectoryPath(traj) {
    while (this.trajectoryGroup.children.length > 0) {
      const child = this.trajectoryGroup.children.pop();
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    }

    const frames = traj.frames;
    if (!frames || frames.length === 0) return;

    // 1. Path line strip
    const points = frames.map((f) => new THREE.Vector3(...f.pose.position));
    const pathGeom = new THREE.BufferGeometry().setFromPoints(points);
    const pathMat = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.8,
    });
    const pathLine = new THREE.Line(pathGeom, pathMat);
    this.trajectoryGroup.add(pathLine);

    // 2. Trajectory points
    const dotMat = new THREE.PointsMaterial({
      color: 0x38bdf8,
      size: 4,
      sizeAttenuation: false,
    });
    const dots = new THREE.Points(pathGeom, dotMat);
    this.trajectoryGroup.add(dots);

    // 3. Scene Center and Bounding Sphere Wireframe
    const center = traj.framing.scene_center;
    const radius = traj.framing.scene_radius;

    const centerMarker = new THREE.AxesHelper(0.35);
    centerMarker.position.set(...center);
    this.trajectoryGroup.add(centerMarker);

    const sphereGeom = new THREE.SphereGeometry(radius, 24, 16);
    const sphereMat = new THREE.MeshBasicMaterial({
      color: 0x64748b,
      wireframe: true,
      transparent: true,
      opacity: 0.22,
    });
    const sphereMesh = new THREE.Mesh(sphereGeom, sphereMat);
    sphereMesh.position.set(...center);
    this.trajectoryGroup.add(sphereMesh);
  }

  updateFramingUI(traj) {
    const framing = traj.framing;
    const centerEl = document.getElementById('framing-scene-center');
    const radiusEl = document.getElementById('framing-scene-radius');
    const activeDistEl = document.getElementById('framing-active-dist');
    const marginNoteEl = document.getElementById('framing-margin-note');

    if (centerEl) centerEl.textContent = `(${framing.scene_center.map((v) => v.toFixed(2)).join(', ')})`;
    if (radiusEl) radiusEl.textContent = `${framing.scene_radius.toFixed(2)} m`;
    if (activeDistEl) activeDistEl.textContent = `${framing.camera_distance.toFixed(2)} m (${framing.framing_mode})`;
    if (marginNoteEl) marginNoteEl.textContent = framing.margin_assumptions;

    // Presets comparison table
    const tbody = document.getElementById('framing-table-body');
    if (tbody && framing.presets_comparison) {
      const fixedD = 6.0;
      const presets = [
        { id: 'historical_fixed', name: 'Historical Fixed (Reference)', occ: `${((framing.scene_radius / (fixedD * Math.tan(Math.PI / 6))) * 100).toFixed(1)}%`, dist: fixedD },
        { id: 'loose', name: 'Adaptive: Loose', occ: '55.0%', dist: framing.presets_comparison.loose },
        { id: 'balanced', name: 'Adaptive: Balanced', occ: '70.0%', dist: framing.presets_comparison.balanced },
        { id: 'tight', name: 'Adaptive: Tight', occ: '85.0%', dist: framing.presets_comparison.tight },
      ];

      tbody.innerHTML = presets
        .map((p) => {
          const isActive = (traj.config.framing_mode === p.id);
          const ratio = ((p.dist / fixedD) * 100).toFixed(1);
          return `
            <tr class="${isActive ? 'row-active' : ''}">
              <td>${p.name} ${isActive ? '◀' : ''}</td>
              <td class="mono">${p.occ}</td>
              <td class="mono">${p.dist.toFixed(2)} m</td>
              <td class="mono">${ratio}%</td>
            </tr>
          `;
        })
        .join('');
    }
  }

  updateDiagnosticsUI(diag, traj = null) {
    const azEl = document.getElementById('diag-az-span');
    const elevEl = document.getElementById('diag-elev-range');
    const radEl = document.getElementById('diag-radius-range');
    const pathEl = document.getElementById('diag-path-length');
    const nearRepEl = document.getElementById('diag-near-repeat');
    const sumNoteEl = document.getElementById('diag-summary-note');

    if (azEl) azEl.textContent = `${diag.azimuth_span_deg.toFixed(1)}°`;
    if (elevEl) elevEl.textContent = `[${diag.elevation_min_deg.toFixed(1)}°, ${diag.elevation_max_deg.toFixed(1)}°]`;
    if (radEl) radEl.textContent = `[${diag.radius_min.toFixed(2)}, ${diag.radius_max.toFixed(2)}] m`;
    if (pathEl) {
      const pLen = diag.trajectory_path_length.toFixed(2);
      const activeTraj = traj || this.currentTrajectory;
      const dur = activeTraj && activeTraj.total_duration_sec !== undefined ? activeTraj.total_duration_sec.toFixed(1) : '-';
      const spd = activeTraj && activeTraj.scientific_speed_m_s !== undefined ? activeTraj.scientific_speed_m_s.toFixed(1) : '1.0';
      pathEl.textContent = `${pLen} m (T=${dur}s @ ${spd} m/s)`;
    }

    if (nearRepEl) {
      if (diag.near_repeat_detected) {
        nearRepEl.innerHTML = `<span style="color:var(--warning);font-weight:600;">Detected</span> (min d=${diag.min_position_distance_m.toFixed(2)}m, min &Delta;&theta;=${diag.min_angular_difference_deg.toFixed(1)}°)`;
      } else {
        nearRepEl.innerHTML = `<span style="color:var(--success);font-weight:600;">None (Smooth 3D coverage)</span> (min d=${diag.min_position_distance_m.toFixed(2)}m, min &Delta;&theta;=${diag.min_angular_difference_deg.toFixed(1)}°)`;
      }
    }
    if (sumNoteEl) sumNoteEl.textContent = diag.diagnostics_summary;
  }

  seekTrajectoryFrame(index, syncElapsed = true) {
    if (!this.currentTrajectory || !this.currentTrajectory.frames.length) return;
    const total = this.currentTrajectory.frames.length;
    const clamped = Math.max(0, Math.min(total - 1, index));
    this.trajectoryFrameIndex = clamped;

    const frame = this.currentTrajectory.frames[clamped];

    if (syncElapsed) {
      const totalDuration = this.currentTrajectory.total_duration_sec || (total / 30);
      if (frame.timestamp_sec !== undefined) {
        this.playbackElapsedSec = frame.timestamp_sec;
      } else {
        this.playbackElapsedSec = (clamped / Math.max(1, total - 1)) * totalDuration;
      }
      this.lastPlaybackTime = null;
    }

    // Update counter and slider
    const counter = document.getElementById('traj-frame-counter');
    if (counter) counter.textContent = `${clamped + 1} / ${total}`;

    const slider = document.getElementById('traj-timeline-slider');
    if (slider && parseInt(slider.value, 10) !== clamped) slider.value = clamped;

    // Update instant radius readout
    const instantRadEl = document.getElementById('framing-instant-radius');
    if (instantRadEl) instantRadEl.textContent = `${frame.distance_to_target.toFixed(2)} m (elev: ${frame.elevation_deg.toFixed(1)}°)`;

    // Update preview quiet meta
    const p = frame.pose;
    const posEl = document.getElementById('traj-meta-pos');
    const lookEl = document.getElementById('traj-meta-look');
    const fovEl = document.getElementById('traj-meta-fov');
    if (posEl) posEl.textContent = `Pos: (${p.position[0].toFixed(1)}, ${p.position[1].toFixed(1)}, ${p.position[2].toFixed(1)})`;
    if (lookEl) lookEl.textContent = `Target: (${p.look_at[0].toFixed(1)}, ${p.look_at[1].toFixed(1)}, ${p.look_at[2].toFixed(1)})`;
    if (fovEl) fovEl.textContent = `FOV: ${p.fov_deg.toFixed(0)}°`;

    // Render camera marker & frustum
    this.render3DCameraPose(frame.pose, 0x38bdf8);

    // Update live observation preview
    this.updateObservationPreview(frame.pose);
  }

  updateObservationPreview(pose) {
    if (!this.previewRenderer || !this.previewCamera) return;

    this.previewCamera.position.set(...pose.position);
    this.previewCamera.up.set(...pose.up);
    this.previewCamera.lookAt(...pose.look_at);
    this.previewCamera.fov = pose.fov_deg;
    this.previewCamera.updateProjectionMatrix();

    // Temporarily hide helper marker & path lines during observation preview
    const markerVis = this.cameraMarkerGroup.visible;
    const trajVis = this.trajectoryGroup.visible;
    const hlVis = this.highlightGroup.visible;

    this.cameraMarkerGroup.visible = false;
    this.trajectoryGroup.visible = false;
    this.highlightGroup.visible = false;

    this.previewRenderer.render(this.scene, this.previewCamera);

    this.cameraMarkerGroup.visible = markerVis;
    this.trajectoryGroup.visible = trajVis;
    this.highlightGroup.visible = hlVis;
  }

  startTrajectoryPlayback() {
    if (this.isTrajectoryPlaying) return;
    if (!this.currentTrajectory || !this.currentTrajectory.frames.length) return;
    this.isTrajectoryPlaying = true;
    const btn = document.getElementById('btn-traj-play');
    if (btn) btn.textContent = '⏸ Pause';

    this.lastPlaybackTime = performance.now();

    const frames = this.currentTrajectory.frames;
    const totalDuration = this.currentTrajectory.total_duration_sec || (frames.length / 30);
    const currFrame = frames[this.trajectoryFrameIndex];
    if (currFrame && currFrame.timestamp_sec !== undefined) {
      this.playbackElapsedSec = currFrame.timestamp_sec;
    } else {
      this.playbackElapsedSec = (this.trajectoryFrameIndex / Math.max(1, frames.length - 1)) * totalDuration;
    }

    const step = (now) => {
      if (!this.isTrajectoryPlaying || this.mode !== 'trajectory') {
        this.pauseTrajectoryPlayback();
        return;
      }
      if (this.lastPlaybackTime === null) {
        this.lastPlaybackTime = now;
      }
      let dtMs = now - this.lastPlaybackTime;
      this.lastPlaybackTime = now;

      // Clamp dt to prevent huge jumps (max 100ms)
      dtMs = Math.min(Math.max(0, dtMs), 100);

      const dtSec = (dtMs / 1000) * this.previewPlaybackRate;
      this.playbackElapsedSec += dtSec;

      if (this.playbackElapsedSec >= totalDuration) {
        this.playbackElapsedSec = this.playbackElapsedSec % totalDuration;
      }

      const frameIdx = this.findFrameIndexForTime(this.playbackElapsedSec);
      if (frameIdx !== this.trajectoryFrameIndex) {
        this.seekTrajectoryFrame(frameIdx, false);
      }

      this.playbackAnimFrame = requestAnimationFrame(step);
    };
    this.playbackAnimFrame = requestAnimationFrame(step);
  }

  findFrameIndexForTime(elapsedSec) {
    if (!this.currentTrajectory || !this.currentTrajectory.frames.length) return 0;
    const frames = this.currentTrajectory.frames;
    if (frames.length === 1) return 0;

    let low = 0;
    let high = frames.length - 1;
    const totalDur = this.currentTrajectory.total_duration_sec || (frames.length / 30);

    while (low <= high) {
      const mid = Math.floor((low + high) / 2);
      const t = frames[mid].timestamp_sec !== undefined
        ? frames[mid].timestamp_sec
        : (mid / (frames.length - 1)) * totalDur;
      if (t <= elapsedSec) {
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }
    return Math.max(0, Math.min(frames.length - 1, high));
  }

  pauseTrajectoryPlayback() {
    this.isTrajectoryPlaying = false;
    this.lastPlaybackTime = null;
    const btn = document.getElementById('btn-traj-play');
    if (btn) btn.textContent = '▶ Play';
    if (this.playbackAnimFrame) {
      cancelAnimationFrame(this.playbackAnimFrame);
      this.playbackAnimFrame = null;
    }
  }

  resetTrajectoryPlaybackClock() {
    this.lastPlaybackTime = null;
    this.playbackElapsedSec = 0.0;
  }

  toggleTrajectoryPlayback() {
    if (this.isTrajectoryPlaying) {
      this.pauseTrajectoryPlayback();
    } else {
      this.startTrajectoryPlayback();
    }
  }

  // =========================================================================
  // Object Inspector
  // =========================================================================
  selectObject(objectIndex, switchTab = true) {
    this.selectedObjectIndex = objectIndex;
    if (!this.currentSnapshot) return;

    this.updateHighlights(this.currentSnapshot);

    const truths = this.currentSnapshot.environment_truth.camera_object_truths;
    const objTruth = truths.find((t) => t.object_index === objectIndex);
    const body = document.getElementById('object-inspector-body');

    if (!objTruth) {
      body.innerHTML = `<div class="inspector-empty-msg">Object index ${objectIndex} not found in camera truths.</div>`;
      return;
    }

    const rawObjs = this.currentSnapshot.environment_truth.objects || this.currentSnapshot.environment_truth.raw_scene_objects || [];
    const rawObj = rawObjs.find((o) => o.object_index === objectIndex) || {};

    this.updateSupportLines(rawObjs, objectIndex);

    const inFrontBadge = objTruth.in_front_of_camera
      ? '<span style="color:var(--success);font-weight:600;">True (depth &gt; 0)</span>'
      : '<span style="color:var(--error);font-weight:600;">False (behind camera)</span>';

    const role = rawObj.role || 'base';
    const placementMode = rawObj.placement_mode || 'on_floor';
    const supportParent = rawObj.support_parent || 'None';
    const bottomZ = Math.max(0, (objTruth.world_location[2] - objTruth.size / 2));
    const elevationStr = `${bottomZ.toFixed(3)} m (center: ${objTruth.world_location[2].toFixed(3)} m)`;
    const rot = rawObj.rotation || [0.0, 0.0, 0.0];
    const rotDegStr = `[${rot[0].toFixed(1)}°, ${rot[1].toFixed(1)}°, ${rot[2].toFixed(1)}°]`;
    const compoundStr = rawObj.compound_id ? `${escapeHtml(rawObj.compound_id)} (part: ${escapeHtml(rawObj.compound_part || 'part')})` : 'None';
    const shapeDisplay = rawObj.shape_variant ? `${escapeHtml(objTruth.shape)} (${escapeHtml(rawObj.shape_variant)})` : escapeHtml(objTruth.shape);

    body.innerHTML = `
      <div class="object-kv-grid">
        <div class="object-kv-row"><span class="k">Object index</span><span class="v mono">${objTruth.object_index}</span></div>
        <div class="object-kv-row"><span class="k">Name</span><span class="v">"${escapeHtml(objTruth.name)}"</span></div>
        <div class="object-kv-row"><span class="k">Shape</span><span class="v">${shapeDisplay}</span></div>
        <div class="object-kv-row"><span class="k">Color</span><span class="v">${escapeHtml(objTruth.color)}</span></div>
        <div class="object-kv-row"><span class="k">Size</span><span class="v mono">${objTruth.size.toFixed(2)} m</span></div>
        <div class="object-kv-row"><span class="k">Role</span><span class="v mono" style="color:var(--accent);font-weight:600;">${escapeHtml(role)}</span></div>
        <div class="object-kv-row"><span class="k">Placement mode</span><span class="v mono">${escapeHtml(placementMode)}</span></div>
        <div class="object-kv-row"><span class="k">Support parent</span><span class="v mono">${escapeHtml(supportParent)}</span></div>
        <div class="object-kv-row"><span class="k">Elevation (floor)</span><span class="v mono">${elevationStr}</span></div>
        <div class="object-kv-row"><span class="k">Rotation (Euler)</span><span class="v mono">${rotDegStr}</span></div>
        <div class="object-kv-row"><span class="k">Compound Part</span><span class="v mono">${compoundStr}</span></div>
        <div class="object-kv-row"><span class="k">World position</span><span class="v mono">(${objTruth.world_location.map((v) => v.toFixed(2)).join(', ')})</span></div>
        <div class="object-kv-row"><span class="k">Camera right</span><span class="v mono">${objTruth.camera_right_x.toFixed(3)} m</span></div>
        <div class="object-kv-row"><span class="k">Camera up</span><span class="v mono">${objTruth.camera_up_y.toFixed(3)} m</span></div>
        <div class="object-kv-row"><span class="k">Camera depth</span><span class="v mono">${objTruth.camera_depth.toFixed(3)} m</span></div>
        <div class="object-kv-row"><span class="k">Metric distance</span><span class="v mono">${objTruth.camera_metric_distance.toFixed(3)} m</span></div>
        <div class="object-kv-row"><span class="k">Front half-space</span><span class="v">${inFrontBadge}</span></div>
      </div>
    `;

    if (switchTab) {
      this.setActiveTab('object');
    }
  }

  // =========================================================================
  // Capability Preflight
  // =========================================================================
  async validateProfilePreflight(profileId) {
    const alertBox = document.getElementById('profile-alert-box');
    alertBox.style.display = 'block';
    alertBox.className = 'profile-alert';
    alertBox.textContent = `Running capability preflight for profile '${profileId}'...`;

    try {
      const res = await fetch(`/api/profiles/validate?profile=${encodeURIComponent(profileId)}`);
      const data = await res.json();
      if (data.status === 'compatible') {
        alertBox.className = 'profile-alert compatible';
        alertBox.textContent = `[PASS] ${data.message} Required: ~${(data.required_memory_estimate_mb / 1024).toFixed(1)} GB.`;
      } else {
        alertBox.className = 'profile-alert incompatible';
        alertBox.textContent = `[INCOMPATIBLE - NO SILENT FALLBACK] ${data.message} Available: ${(data.available_memory_mb / 1024).toFixed(1)} GB, Required: ${(data.required_memory_estimate_mb / 1024).toFixed(1)} GB. Recommendation: ${data.recommended_profile || 'None'}.`;
      }
    } catch (e) {
      alertBox.className = 'profile-alert incompatible';
      alertBox.textContent = `Preflight error: ${e.message}`;
    }
  }

  // =========================================================================
  // Telemetry Polling
  // =========================================================================
  startTelemetryPolling() {
    setInterval(async () => {
      try {
        let url = '/api/telemetry';
        if (this.currentRunId) url += `?run_id=${encodeURIComponent(this.currentRunId)}`;
        const res = await fetch(url);
        const data = await res.json();

        const dotEl = document.getElementById('telem-status-dot');
        const freshEl = document.getElementById('telem-freshness');
        const freshness = data.freshness || 'missing';

        if (dotEl) {
          dotEl.className = `status-dot status-dot-${freshness}`;
        }
        if (freshEl) {
          freshEl.textContent = freshness.charAt(0).toUpperCase() + freshness.slice(1);
        }

        document.getElementById('telem-run').textContent = this.currentRunId || 'None';

        if (data.progress) {
          const p = data.progress;
          document.getElementById('telem-samples').textContent = `${p.processed_samples || '-'} / ${p.total_samples || '-'}`;
          document.getElementById('telem-steps').textContent = `${p.optimizer_step || '-'} / ${p.total_optimizer_steps || '-'}`;
          document.getElementById('telem-loss').textContent = p.latest_raw_loss !== undefined ? p.latest_raw_loss.toFixed(4) : '-';
          document.getElementById('telem-sps').textContent = p.samples_per_sec ? `${p.samples_per_sec.toFixed(2)} samples/s` : '- samples/s';
        }

        if (data.telemetry && data.telemetry.nvml) {
          const n = data.telemetry.nvml;
          if (n.gpu_utilization_pct !== undefined) {
            document.getElementById('telem-gpu-util').textContent = `${n.gpu_utilization_pct.toFixed(0)} %`;
          }
          if (n.nvml_gpu_memory_used_mb !== undefined) {
            document.getElementById('telem-vram').textContent = `${(n.nvml_gpu_memory_used_mb / 1024).toFixed(1)} / ${(n.nvml_gpu_memory_total_mb / 1024).toFixed(1)} GB`;
          }
          if (n.nvml_gpu_memory_total_mb !== undefined) {
            const availEl = document.getElementById('spec-avail');
            if (availEl) {
              const freeMb = n.nvml_gpu_memory_total_mb - (n.nvml_gpu_memory_used_mb || 0);
              availEl.textContent = `${(freeMb / 1024).toFixed(1)} GB free / ${(n.nvml_gpu_memory_total_mb / 1024).toFixed(1)} GB`;
            }
          }
          if (n.gpu_power_w !== undefined) {
            document.getElementById('telem-power').textContent = `${n.gpu_power_w.toFixed(0)} W`;
          }
          if (n.sm_clock_mhz !== undefined) {
            document.getElementById('telem-clock').textContent = `${n.sm_clock_mhz.toFixed(0)} MHz`;
          }
        }
      } catch (e) {
        // Silently skip transient polling glitches
      }
    }, 1000);
  }
}

// Bootstrap application on DOM load
window.addEventListener('DOMContentLoaded', () => {
  new GodViewApp();
});

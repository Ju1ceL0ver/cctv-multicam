"""Running the v2 model on video: world people over both cameras, permanent staff, answers now and after a delay.

Per moment (both cameras' frames of one tick):
  1. the model: maps, slots (proposals + this camera's tracks), the cross-camera layer;
  2. a camera's slot becomes (or stays) a track when its person-ness >= KEEP, or while the model calls it hidden
     (state) for up to HIDDEN_S; 'gone' ends it;
  3. world people: a track born this moment is given to a world person by SameHead -- against everybody the
     world remembers (either camera, any time today; staff never forgotten) -- Hungarian, with 'nobody' at the
     threshold; the two cameras' tracks of this moment are joined the same way (same moment, floor distance);
  4. answers: the provisional one at once, the final one DELAY seconds later (v2_lag.py, when trained; until
     then the final one is the provisional one read DELAY later).

Anonymous numbers only: a world person is a number and vectors, no picture is kept."""
import numpy as np

KEEP = 0.5            # a new person: a proposal this sure
KEEP_TRACK = 0.25     # an existing track goes on at this (ByteTrack: a half-hidden person scores low but is still there)
HIDDEN_S = 25.0
SAME = 0.5
DUP_IOU = 0.5         # two slots whose masks overlap this much are one person (the v1 finding: 9x fewer false ones)
DUP_TRACK_IOU = 0.85  # two tracks this much on one person are merged (the v0 tracker's RA_DUPIOU) ...
DUP_TRACK_APP = 0.35  # ... if they look alike too (RA_DUPAPP: two people side by side are not merged)
CROSS_CAMERA = False  # joining the cameras' people waits for SameHead to learn camera-to-camera pairs
TICK = 0.08


class World:
    def __init__(self, staff=()):
        self.people = {}              # id -> {'ident': mean vector, 'n', 'xy', 'var', 't', 'cam', 'staff'}
        self.next = 1
        self.staff = set(staff)

    def new(self, ident, xy, var, t, cam):
        i = self.next
        self.next += 1
        self.people[i] = {'ident': ident.copy(), 'n': 1, 'xy': xy, 'var': var, 't': t, 'cam': cam, 'staff': i in self.staff}
        return i

    def see(self, i, ident, xy, var, t, cam):
        p = self.people[i]
        k = min(p['n'], 50)
        p['ident'] = (p['ident'] * k + ident) / (k + 1)
        p['n'] += 1
        p.update(xy=xy, var=var, t=t, cam=cam)


class Runtime:
    def __init__(self, model, dev, staff=(), delay_s=3.0):
        import torch
        import v2_data as VD
        self.model, self.dev = model.eval(), dev
        self.asm = VD.Assembler(dev, getattr(model, 'rgb_size', None))
        self.tracks = [None, None]
        self.track_world = [[], []]           # world id of each track slot
        self.hidden_since = [[], []]
        self.scene = [None, None]
        self.world = World(staff)
        self.delay = int(round(delay_s / TICK))
        self.buffer = []
        self.torch = torch

    def _same(self, a_ident, b_ident, ctx):
        torch = self.torch
        with torch.no_grad():
            return torch.sigmoid(self.model.same(torch.as_tensor(a_ident, device=self.dev).float(),
                                                 torch.as_tensor(b_ident, device=self.dev).float(),
                                                 torch.as_tensor(ctx, device=self.dev).float())).cpu().numpy()

    def assign_new(self, idents, xys, vars_, t, cam, taken):
        """New tracks -> world people (or new ones)."""
        from scipy.optimize import linear_sum_assignment
        ids = [i for i in self.world.people if i not in taken and (CROSS_CAMERA or self.world.people[i]['cam'] == cam)]
        if not len(idents):
            return []
        if not ids:
            return [self.world.new(idents[k], xys[k], vars_[k], t, cam) for k in range(len(idents))]
        P = np.zeros((len(idents), len(ids)))
        for a in range(len(idents)):
            A = np.repeat(idents[a][None], len(ids), 0)
            B = np.stack([self.world.people[i]['ident'] for i in ids])
            ctx = []
            for i in ids:
                p = self.world.people[i]
                dt = (t - p['t']) * TICK / 60
                dist = float(np.linalg.norm(np.asarray(xys[a]) - np.asarray(p['xy'])))
                var = float(np.sqrt(vars_[a] + p['var']))
                ctx.append([dt, dist, var, float(p['cam'] == cam), 1.0, 1.0, 1.0, 1.0])
            P[a] = self._same(A, B, np.array(ctx))
        r, c = linear_sum_assignment(-P)
        out = [None] * len(idents)
        for a, j in zip(r, c):
            if P[a, j] >= SAME:
                out[a] = ids[j]
                self.world.see(ids[j], idents[a], xys[a], vars_[a], t, cam)
        for a in range(len(idents)):
            if out[a] is None:
                out[a] = self.world.new(idents[a], xys[a], vars_[a], t, cam)
        return out

    def step(self, t, frames, masks_out=False):
        """frames: [ {img, bg_long, bg_now} for cam1, cam2 ] of tick t -> provisional answer; returns also the
        final answer of tick t - delay when due."""
        torch = self.torch
        import slot_v2 as V
        import train_v2 as TV
        model = self.model
        with torch.no_grad(), torch.autocast('cuda', torch.bfloat16, enabled=self.dev == 'cuda'):
            rgb, bgv, sta, cam_ids = self.asm(frames, ('cam1', 'cam2'), False)
            bg_sem = model.body.vit(bgv)[0]
            m = model.maps(rgb, sta, cam_ids, bg_sem, self.asm.last_world)
            tb = model.stack_tracks(self.tracks, self.dev, m['mem'].dtype)
            sc = torch.cat([(self.scene[b] if self.scene[b] is not None else model.scene.start(cam_ids[b:b + 1])) for b in range(2)], 0)
            r = model.decode(m, cam_ids, tb, sc)
            r1, r2 = model.cross_cameras(TV.split_b(r, 0), TV.split_b(r, 1))
        answer = {'t': t, 'people': []}
        for b, r in ((0, r1), (1, r2)):
            mb = TV.split_b(m, b)
            p = r['obj'][0].float().sigmoid().cpu().numpy()
            pad = r['pad'][0].cpu().numpy()
            state = r['state'][0].float().softmax(-1).cpu().numpy()
            place = r['place'][0].float().cpu().numpy()
            T = r['tracks']
            ntr = len(self.track_world[b])
            with torch.no_grad(), torch.autocast('cuda', torch.bfloat16, enabled=self.dev == 'cuda'):
                logits = model.full_masks(mb, r)[0]
                ident = model.pooled(mb, r, logits.float().sigmoid()[None].to(logits.dtype))['ident'][0].float().cpu().numpy()
            masks = (logits > 0).cpu().numpy()
            area = masks.reshape(len(masks), -1).sum(1)
            keep = np.zeros(len(p), bool)
            world = [None] * len(p)
            for s_ in range(min(T, ntr)):                        # tracks: on at a lower score, or hidden a while
                hidden = state[s_, 1] > max(state[s_, 0], state[s_, 2]) and (t - self.hidden_since[b][s_]) * TICK < HIDDEN_S
                if p[s_] >= KEEP_TRACK or hidden:
                    keep[s_] = True; world[s_] = self.track_world[b][s_]
            new = [s_ for s_ in range(T, len(p)) if not pad[s_] and p[s_] >= KEEP and area[s_] > 0]
            # duplicates: a slot whose mask lies on a kept one (tracks first, then the surer) is the same person
            order = [s_ for s_ in range(len(p)) if keep[s_] and p[s_] >= KEEP_TRACK] + sorted(new, key=lambda s_: -p[s_])
            kept_masks = []
            for s_ in order:
                inter = [np.logical_and(masks[s_], masks[q]).sum() / max(1, np.logical_or(masks[s_], masks[q]).sum()) for q in kept_masks]
                if inter and max(inter) >= DUP_IOU:
                    q = kept_masks[int(np.argmax(inter))]
                    if s_ < T and q < T and max(inter) >= DUP_TRACK_IOU and self._cos(ident[s_], ident[q]) >= 1 - DUP_TRACK_APP:
                        keep[s_] = False                           # two tracks on one person: the younger one ends
                    elif s_ >= T:
                        continue                                   # a proposal on a tracked person: not new
                    else:
                        kept_masks.append(s_)
                        continue
                    continue
                kept_masks.append(s_)
            new = [s_ for s_ in new if s_ in kept_masks]
            xy = place[:, :2] * V.XY_S + np.array(V.XY_C)
            var = np.exp(place[:, 3:5]).sum(1) * V.XY_S ** 2
            taken = {w for w in world if w is not None}
            got = self.assign_new([ident[s_] for s_ in new], [xy[s_] for s_ in new], [var[s_] for s_ in new], t, b, taken)
            for s_, w in zip(new, got):
                keep[s_] = True; world[s_] = w
            for s_ in range(len(p)):
                if keep[s_] and world[s_] is not None and p[s_] >= KEEP_TRACK and s_ in kept_masks:
                    self.world.see(world[s_], ident[s_], xy[s_], var[s_], t, b)
                    person = {'cam': b, 'world': world[s_], 'score': float(p[s_]), 'xy': xy[s_].tolist(),
                              'z': float(place[s_, 2]), 'state': int(state[s_].argmax()),
                              'box': r['box'][0, s_].float().cpu().tolist(), 'staff': self.world.people[world[s_]]['staff']}
                    if masks_out:
                        person['mask'] = masks[s_]
                    answer['people'].append(person)
            kt = torch.as_tensor(keep, device=self.dev)[None]
            prev = None if tb is None else {k_: v[b:b + 1] for k_, v in tb.items()}
            with torch.no_grad(), torch.autocast('cuda', torch.bfloat16, enabled=self.dev == 'cuda'):
                self.tracks[b], idx = model.next_tracks(r, kt, prev)
                self.scene[b] = model.scene.update(sc[b:b + 1], model.norm(r['q']), r['pad'])
            kept = idx[0].tolist()
            prev_hidden = self.hidden_since[b]
            self.track_world[b] = [world[s_] for s_ in kept]
            self.hidden_since[b] = [(prev_hidden[s_] if s_ < ntr and p[s_] < KEEP_TRACK else t) for s_ in kept]
        self.buffer.append(answer)
        final = self.buffer.pop(0) if len(self.buffer) > self.delay else None
        return answer, final

    @staticmethod
    def _cos(a, b):
        return float(np.dot(a, b) / max(1e-8, np.linalg.norm(a) * np.linalg.norm(b)))

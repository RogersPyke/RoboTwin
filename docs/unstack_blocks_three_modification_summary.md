# Unstack Blocks Three: Modification Summary

## English

### 1. What We Built

- **Task**: Three-block unstack with **random place targets** (random position and random orientation per episode).
- **Setup**: Three blocks stacked at center: block1 (red, bottom), block2 (green, middle), block3 (blue, top). Stack center `(0, -0.13)`, heights `z_base`, `z_base+0.05`, `z_base+0.10`.
- **Execution**: Unstack in **top-first order** (block3 → block2 → block1). Each block is grasped from above, lifted, then placed at its target pose. Targets are sampled in `load_actors`: random (x, y) in `TARGET_XLIM` / `TARGET_YLIM`, random yaw in `[-pi, pi]`, fixed z and upright.
- **Success**: All three blocks on table, grippers open.

---

### 2. Solving “No Feasible Solution” in the Placing Stage (Obstacle on Ground)

**Problem**: During the **placing** phase, when the arm carries one block to its target, other block(s) already on the table act as **obstacles**. The motion planner (optimization) must find a collision-free path from current pose to the place pose. With default or lateral approaches, the planned path often **passed through the obstacle region** (the block(s) on the ground), so the optimizer returned **no feasible solution** and the task failed in many seeds.

**What actually fixed it**:

1. **Top-down grasp** (`contact_point_id=[0, 1, 2, 3]`): The arm approaches the stack **straight from above** and lifts **vertically**. It does not sweep sideways through the table plane where the remaining block(s) sit. So the grasp/retreat motion stays in a narrow vertical column over the stack and avoids the obstacle.
2. **Top-down place** (`pre_dis_axis="fp"` in `place_actor`): The place approach is along the **target pose’s z-axis** (i.e. straight down onto the target). The arm goes to a pre-place pose above the target, then moves straight down to place. So the **placing motion is also vertical**: it stays in a vertical tube above the target and does not pass through the table-level area where the obstacle block(s) are. That greatly reduces the chance of the path intersecting the obstacle, so the planner often finds a **feasible** path.

**Takeaway**: Constraining the motion to **vertical approach and vertical retreat** (both at grasp and at place) keeps the swept volume away from the obstacle blocks on the ground. That is what made the optimization/planning process return a feasible solution consistently, instead of failing due to the obstacle block.

**Waypoint (and why we did not keep it)**: We also tried adding a **waypoint**: first move to a high point above the stack and midway between the stack and the target, then go to the place pose. The idea was to “fly over” the obstacle. In practice, **the waypoint did not always have a positive effect**: it sometimes helped, but sometimes did not improve feasibility or even made planning harder (e.g. extra segment to plan, or a worse local minimum). So we **removed the waypoint** and relied on **top-down-only motion** (grasp + place both vertical) as the main solution. The lesson: constraining the motion primitive (strict top-down) was the reliable fix; adding a waypoint is not a silver bullet and can be counterproductive.

---

### 3. What Made the Modifications Succeed (Detailed)

#### 3.1 Top-Down Grasp Only

- **What**: In `unstack_and_place_block`, call `grasp_actor` with **`contact_point_id=[0, 1, 2, 3]`**. These indices are the top-face contact points of the box (see `create_actor` / contact point definition).
- **Why**: With only top-face points, `choose_grasp_pose` in the base task considers only approach-from-above grasps. The arm then moves straight down to the top of the block and lifts vertically, so it does not sweep through the space of the block(s) still on the table. Without this, the “obstacle” block left on the stack made many seeds fail (planning or collision).
- **Where**: `envs/unstack_blocks_three.py`, `grasp_kw` in `unstack_and_place_block`.

#### 3.2 Gripper Standoff (Avoid Deep / Double-Grasp)

- **What**: Use **`grasp_dis=GRASP_MIN_STANDOFF`** (e.g. `0.02` m) in `grasp_actor`. In the base task, the final grasp pose is computed as `pre_pose + (pre_grasp_dis - grasp_dis)` along the approach direction. So a positive `grasp_dis` keeps the gripper that much **further from the block surface** at close.
- **Why**: With `grasp_dis=0`, the gripper is driven all the way to the nominal contact surface and often penetrated too deep, causing double-grasp or unstable contact. A small standoff (e.g. 2 cm) keeps a minimum clearance and gives reliable, repeatable grasps.
- **Constants**: `GRASP_MIN_STANDOFF = 0.02` at top of `unstack_blocks_three.py`.

#### 3.3 Place Orientation Actually Used (The Main Fix)

- **What**: When calling **`place_actor`**, pass **`constrain="align"`** and **`align_axis=None`** (in addition to `target_pose`, `pre_dis_axis="fp"`, etc.).
- **Why**:
  - **Default behaviour**: The base `get_place_pose` uses `constrain="auto"` when not overridden. In that branch it never uses the **target pose’s orientation**. It chooses fixed world align axes (e.g. `[1,1,0]`, `[-1,1,0]`, or `[0,1,0]` depending on arm and grasp direction). So the target’s quaternion was ignored and random orientations had no effect.
  - **With `constrain="align"` and `align_axis=None`**: The base passes these to `transforms.get_place_pose`. There, when `align_axis is None`, it is set to the **target pose’s x-axis** (`target_pose_mat[:3, :3] @ [1,0,0]`). The placed block’s position is set from the target; its z is aligned to the target z (upright); then its x is aligned to the target x. So the **full target orientation** (including random yaw) is applied.
- **Where**: `envs/unstack_blocks_three.py`, `place_actor(..., constrain="align", align_axis=None)`. Under the hood: `_base_task.get_place_pose` → `transforms.get_place_pose` with `constrain="align"`, `align_axis=None`.

#### 3.4 Random Target Pose Construction

- **Position**: For each of the three targets, sample (x, y) from `TARGET_XLIM` and `TARGET_YLIM`. Reject samples that are too close to the stack center (distance &lt; `MIN_TARGET_TO_STACK`) or too close to each other (pairwise distance &lt; `MIN_TARGET_SEP`). Up to 200 attempts; on failure use fallback poses. z is fixed: `z_t = 0.75 + self.table_z_bias`.
- **Orientation**: Block stays upright; only **yaw** is randomized. Base orientation is `BASE_QUAT = [0, 1, 0, 0]`. For each target, sample `yaw = np.random.uniform(-np.pi, np.pi)` and set `q = qmult(BASE_QUAT, euler2quat(0, 0, yaw))` (transforms3d). So the target pose is `[x, y, z_t, *q]` in (x, y, z, qw, qx, qy, qz) form.
- **Constants**: `TARGET_XLIM`, `TARGET_YLIM`, `MIN_TARGET_SEP`, `MIN_TARGET_TO_STACK`, `BASE_QUAT` in `unstack_blocks_three.py`.

---

### 4. Key Takeaway (English)

- **Feasibility (obstacle on ground)**: The fix for “no feasible solution” in the placing stage was **strict top-down motion** (grasp and place both vertical), so the swept volume avoids the obstacle block(s). A waypoint was tried but **did not always help** and was removed; constraining the motion primitive was the reliable approach.
- **Orientation**: The fix for random orientation taking effect was passing **`constrain="align"`** and **`align_axis=None`** into `place_actor`, so the framework uses the target’s full rotation instead of fixed world axes.

---

---

## 中文

### 1. 实现内容

- **任务**：三块积木拆垛，且**放置目标随机**（每局随机位置与随机朝向）。
- **场景**：三块积木在桌面中心叠放：block1（红，底）、block2（绿，中）、block3（蓝，顶）。堆叠中心 `(0, -0.13)`，高度分别为 `z_base`、`z_base+0.05`、`z_base+0.10`。
- **执行**：按**从顶到底**顺序拆放（block3 → block2 → block1）。每块从正上方抓取、抬起，再放到对应目标位姿。目标在 `load_actors` 中采样：在 `TARGET_XLIM` / `TARGET_YLIM` 内随机 (x, y)，在 `[-pi, pi]` 内随机 yaw，z 固定，保持直立。
- **成功条件**：三块均置于桌面上且夹爪张开。

---

### 2. 放置阶段“无可行解”的成因与解决（地面障碍块）

**问题**：在**放置**阶段，机械臂将当前块运向目标时，桌上已有其他块作为**障碍**。运动规划（优化）需要求出一条从当前位姿到放置位姿的无碰撞路径。若采用默认或侧向接近方式，规划路径往往会**穿过障碍区域**（地面上的块），导致优化器**无可行解**，任务在多颗种子上失败。

**实际有效的做法**：

1. **自上而下抓取**（`contact_point_id=[0, 1, 2, 3]`）：机械臂**从正上方**接近堆叠并**竖直**提起，不会在桌面平面内横向扫过仍留在桌上的块。因此抓取/撤离运动仅占用堆叠正上方的一条竖直通道，避开障碍。
2. **自上而下放置**（`place_actor` 中 `pre_dis_axis="fp"`）：放置接近方向沿**目标位姿的 z 轴**（即正对目标竖直下降）。机械臂先到目标上方的预放置位姿，再竖直下降完成放置。因此**放置运动也是竖直的**：仅占用目标正上方的竖直通道，不会经过障碍块所在的桌面高度区域，路径与障碍相交的概率大大降低，规划器更容易得到**可行**解。

**结论**：将运动约束为**竖直接近、竖直离开**（抓取与放置均如此），使扫过的体积避开地面上的障碍块，这是优化/规划过程能稳定得到可行解、而非因障碍块报无解的关键。

**关于途经点（waypoint）及为何未保留**：我们曾尝试增加**途经点**：先移动到堆叠与目标之间中线上方的高点，再前往放置位姿，意图“越过”障碍。实践中**途经点并不总能带来正面效果**：有时有帮助，有时对可行性无改善甚至使规划更困难（例如多一段需规划的轨迹，或陷入更差的局部）。因此我们**移除了途经点**，仅依靠**纯自上而下运动**（抓取与放置均为竖直）作为主要方案。教训：约束运动基元（严格自上而下）是可靠手段；增加途经点并非万能，有时反而不利。

---

### 3. 修改成功的关键（详细说明）

#### 3.1 仅允许自上而下抓取

- **做法**：在 `unstack_and_place_block` 中调用 `grasp_actor` 时传入 **`contact_point_id=[0, 1, 2, 3]`**。这些索引对应盒子**顶面**的接触点（见 `create_actor` / 接触点定义）。
- **原因**：只使用顶面接触点时，基类中的 `choose_grasp_pose` 只会考虑“从正上方接近”的抓取。机械臂沿竖直方向接近块顶并竖直提起，不会扫过仍留在桌上的其他块所在空间。若不限制，留在堆上的“障碍”块会导致大量种子规划失败或发生碰撞。
- **位置**：`envs/unstack_blocks_three.py` 中 `unstack_and_place_block` 的 `grasp_kw`。

#### 3.2 夹爪与块面保持距离（避免“插得过深”/二次夹持）

- **做法**：在 `grasp_actor` 中使用 **`grasp_dis=GRASP_MIN_STANDOFF`**（如 `0.02` 米）。基类中最终抓取位姿由 `pre_pose + (pre_grasp_dis - grasp_dis)` 沿接近方向得到；因此正的 `grasp_dis` 使夹爪在闭合时与块表面**保持一段距离**。
- **原因**：`grasp_dis=0` 时夹爪会贴到名义接触面，容易插得过深，造成二次夹持或不稳定接触。约 2 cm 的预留使块面与夹爪保持最小间隙，抓取更稳定、可重复。
- **常量**：`unstack_blocks_three.py` 顶部 `GRASP_MIN_STANDOFF = 0.02`。

#### 3.3 放置时真正使用目标朝向（核心修改）

- **做法**：调用 **`place_actor`** 时显式传入 **`constrain="align"`** 和 **`align_axis=None`**（与 `target_pose`、`pre_dis_axis="fp"` 等一起）。
- **原因**：
  - **默认行为**：基类 `get_place_pose` 在未覆盖时使用 `constrain="auto"`。该分支下**从不使用目标位姿的朝向**，而是根据手臂和抓取方向选用固定的世界对齐轴（如 `[1,1,0]`、`[-1,1,0]` 或 `[0,1,0]`）。因此目标的四元数被忽略，随机朝向不起作用。
  - **使用 `constrain="align"` 且 `align_axis=None`**：基类将这两个参数传给 `transforms.get_place_pose`。其中当 `align_axis is None` 时，会取**目标位姿的 x 轴**（`target_pose_mat[:3, :3] @ [1,0,0]`）。放置时物体位置取自目标；z 轴与目标 z 对齐（直立）；再令物体 x 轴与目标 x 轴对齐。因此**目标的完整朝向**（含随机 yaw）被应用。
- **位置**：`envs/unstack_blocks_three.py` 中 `place_actor(..., constrain="align", align_axis=None)`。调用链：`_base_task.get_place_pose` → `transforms.get_place_pose`，且传入 `constrain="align"`、`align_axis=None`。

#### 3.4 随机目标位姿的构造

- **位置**：三个目标各自在 `TARGET_XLIM`、`TARGET_YLIM` 内采样 (x, y)。若某样本距堆叠中心过近（&lt; `MIN_TARGET_TO_STACK`）或两两距离过近（&lt; `MIN_TARGET_SEP`）则舍弃。最多尝试 200 次；失败则用备用位姿。z 固定为 `z_t = 0.75 + self.table_z_bias`。
- **朝向**：块保持直立，仅**绕竖直轴 yaw** 随机。基础朝向 `BASE_QUAT = [0, 1, 0, 0]`。对每个目标采样 `yaw = np.random.uniform(-np.pi, np.pi)`，并设 `q = qmult(BASE_QUAT, euler2quat(0, 0, yaw))`（transforms3d）。目标位姿格式为 `[x, y, z_t, *q]`，即 (x, y, z, qw, qx, qy, qz)。
- **常量**：`unstack_blocks_three.py` 中的 `TARGET_XLIM`、`TARGET_YLIM`、`MIN_TARGET_SEP`、`MIN_TARGET_TO_STACK`、`BASE_QUAT`。

---

### 4. 核心结论（中文）

- **可行性（地面障碍）**：解决放置阶段“无可行解”的做法是**严格自上而下运动**（抓取与放置均为竖直），使扫过体积避开障碍块。途经点（waypoint）曾尝试但**并不总能改善**，已移除；约束运动基元才是可靠手段。
- **朝向**：让随机朝向生效的做法是在调用 `place_actor` 时传入 **`constrain="align"`** 和 **`align_axis=None`**，使框架按目标的完整旋转对齐，而非固定世界轴。

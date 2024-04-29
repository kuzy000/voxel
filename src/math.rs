use std::ops::Mul;
use bevy::prelude::*;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct IMat4 {
    pub x_axis: IVec4,
    pub y_axis: IVec4,
    pub z_axis: IVec4,
    pub w_axis: IVec4,
}

impl IMat4 {
    pub const ZERO: Self = Self::from_cols(IVec4::ZERO, IVec4::ZERO, IVec4::ZERO, IVec4::ZERO);
    pub const IDENTITY: Self = Self::from_cols(IVec4::X, IVec4::Y, IVec4::Z, IVec4::W);

    pub const fn from_cols(x_axis: IVec4, y_axis: IVec4, z_axis: IVec4, w_axis: IVec4) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
            w_axis,
        }
    }

    pub fn mul_vec4(&self, rhs: IVec4) -> IVec4 {
        let mut res = self.x_axis * rhs.xxxx();
        res = res + self.y_axis * rhs.yyyy();
        res = res + self.z_axis * rhs.zzzz();
        res = res + self.w_axis * rhs.wwww();
   
        res
    }
    
    pub fn mul_mat4(&self, rhs: &Self) -> Self {
        Self::from_cols(
            self.mul_vec4(rhs.x_axis),
            self.mul_vec4(rhs.y_axis),
            self.mul_vec4(rhs.z_axis),
            self.mul_vec4(rhs.w_axis),
        )
    }

    pub fn col_mut(&mut self, index: usize) -> &mut IVec4 {
        match index {
            0 => &mut self.x_axis,
            1 => &mut self.y_axis,
            2 => &mut self.z_axis,
            3 => &mut self.w_axis,
            _ => panic!("index out of bounds"),
        }
    }

    pub fn from_translation(translation: IVec3) -> Self {
        Self::from_cols(
            IVec4::X,
            IVec4::Y,
            IVec4::Z,
            IVec4::new(translation.x, translation.y, translation.z, 1),
        )
    }
}

impl Default for IMat4 {
    #[inline]
    fn default() -> Self {
        return IMat4::IDENTITY;
    }
}

impl Mul<IVec4> for IMat4 {
    type Output = IVec4;

    #[inline]
    fn mul(self, rhs: IVec4) -> Self::Output {
        self.mul_vec4(rhs)
    }
}

impl Mul<IMat4> for IMat4 {
    type Output = IMat4;

    #[inline]
    fn mul(self, rhs: IMat4) -> Self::Output {
        self.mul_mat4(&rhs)
    }
}
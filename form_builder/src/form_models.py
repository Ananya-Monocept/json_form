from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class IValidator(BaseModel):
    validatorName: Optional[str] = None
    message: Optional[str] = None
    required: Optional[bool] = None
    pattern: Optional[str] = None
    minLength: Optional[int] = None
    maxLength: Optional[int] = None
    email: Optional[str] = None

class IOptions(BaseModel):
    id: Optional[str]
    name: Optional[str]
    other: Optional[Any]
    value: Optional[Any]
    class_: Optional[str]  # Renamed to avoid conflict with Python's `class`
    selected: Optional[bool]
    dependentControls: Optional[List[str]]
    disabled: Optional[bool]

class IRadioOption(BaseModel):
    name: str
    label: str
    value: Any
    selected: Optional[bool]
    year: Optional[str]
    discount: Optional[str]
    dependentControls: Optional[List[str]]
    visible: bool

class ISelectCheckboxOption(BaseModel):
    label: Optional[str]
    value: str
    button: Optional[bool]
    imagePath: Optional[str]
    isIncrement: Optional[bool]
    name: Optional[str]
    dependentControls: Optional[Any]
    gender: Optional[Any]
    id: Optional[Any]
    memberRelationCode: Optional[Any]
    productId: Optional[Any]

class IImage(BaseModel):
    id: int
    src: str
    alt: str
    width: Optional[str]
    height: Optional[str]
    label: str

class IAdditionalQuestionOption(BaseModel):
    label: str
    value: bool
    type_: Optional[str]  # Renamed to avoid conflict with Python's `type`

class IAdditionalQuestion(BaseModel):
    id: Optional[int]
    label: str
    type_: Optional[str]  # Renamed to avoid conflict with Python's `type`
    options: Optional[List[Any]]  # Can include strings or IAdditionalQuestionOption
    selectedOption: Optional[str]

class IAdditionalCover(BaseModel):
    id: int
    value: str
    class_: str  # Renamed to avoid conflict with Python's `class`
    selected: bool
    description: str
    additionalQuestions: Optional[List[IAdditionalQuestion]]

class IDynamicControl(BaseModel):
    name: str
    label: str
    visibleLabel: bool
    key: Optional[str]
    type_: Optional[str]  # Renamed to avoid conflict with Python's `type`
    value: Optional[Any]
    apiEndpoint: Optional[str]
    disabled: Optional[bool]
    relationDisabled: Optional[bool]
    questionCondition: Optional[bool]
    class_: Optional[str]  # Renamed to avoid conflict with Python's `class`
    restrictKeyPress: Optional[bool]
    methodName: Optional[str]
    visible: Optional[bool]
    options: Optional[List[IOptions]]
    validators: Optional[List[IValidator]]
    radioOptions: Optional[List[IRadioOption]]
    selectCheckboxOptions: Optional[List[ISelectCheckboxOption]]
    bigFont: Optional[bool]
    subControls: Optional[List[List["ISubControl"]]]
    innerArrayControl: Optional[List[List["IDynamicControl"]]]
    innerControls: Optional[List["ISubControl"]]
    innerSubControls: Optional[List["ISubControl"]]
    image: Optional[IImage]
    tabs: Optional[List["ITab"]]
    onChangeMethod: Optional[str]
    getAllOption: Optional[str]
    maxDateLength: Optional[Any]
    minDateLength: Optional[Any]
    maxLength: Optional[int]
    minLength: Optional[int]
    inputMaxLength: Optional[int]

class ISubControl(BaseModel):
    name: str
    visibleLabel: Optional[bool]
    label: Optional[str]
    class_: Optional[str]  # Renamed to avoid conflict with Python's `class`
    type_: Optional[str]  # Renamed to avoid conflict with Python's `type`
    validators: Optional[List[IValidator]]
    value: Optional[Any]
    radioOptions: Optional[List[IRadioOption]]
    selectCheckboxOptions: Optional[List[ISelectCheckboxOption]]
    options: Optional[List[IOptions]]
    visible: Optional[bool]
    disabled: Optional[bool]
    method: Optional[str]
    innerControls: Optional[List["ISubControl"]]
    innerSubControls: Optional[List["ISubControl"]]
    displayOnly: Optional[bool]
    coreControls: Optional[List["ISubControl"]]
    extraBenefitsControls: Optional[List["ISubControl"]]
    bigFont: Optional[bool]
    dependentControls: Optional[List[str]]
    getAllOption: Optional[str]
    isButton: Optional[bool]
    innerArrayControl: Optional[List[List["IDynamicControl"]]]
    conditionCheck: Optional[bool]
    maxDateLength: Optional[Any]
    minDateLength: Optional[Any]
    maxLength: Optional[int]
    visibleToolTip: Optional[bool]
    toolTipMessage: Optional[str]
    onChangeMethod: Optional[str]
    methodName: Optional[str]
    allowedRelations: Optional[List[str]]
    tooolTipTable: Optional[List[Any]]
    visibleToolTipTable: Optional[bool]
    tableHeadArray: Optional[List[Any]]

class ITab(BaseModel):
    name: str
    label: str
    content: Any
    selectCheckboxOptions: Optional[List[ISelectCheckboxOption]]
    type_: str  # Renamed to avoid conflict with Python's `type`
    class_: str  # Renamed to avoid conflict with Python's `class`
    visibleLabel: bool

class IConditionalVisibility(BaseModel):
    dependsOn: str
    values: List[Any]

class IFormControl(BaseModel):
    name: str
    label: Optional[str] = None
    visibleLabel: Optional[bool] = True
    bigFont: Optional[bool] = None
    content: Optional[List[IOptions]] = None
    text: Optional[str] = None
    value: Optional[Any] = None
    apiEndpoint: Optional[str] = None
    method: Optional[str] = None
    controlTypeName: Optional[str] = None
    options: Optional[List[IOptions]] = None
    idProperty: Optional[str] = None
    nameProperty: Optional[str] = None
    restrictKeyPress: Optional[bool] = None
    class_: Optional[str] = None
    cssClass: Optional[str] = None
    showBorder: Optional[bool] = None
    type_: Optional[str] = None
    subType: Optional[str] = None
    variableName: Optional[str] = None
    validators: Optional[List[IValidator]] = None
    validationRules: Optional[List[Any]] = None
    disabled: Optional[bool] = None
    dynamicControls: Optional[List[List[IDynamicControl]]] = None
    innerArrayControl: Optional[List[List[IDynamicControl]]] = None
    bannerText: Optional[str] = None
    image: Optional[IImage] = None
    additionalCovers: Optional[List[IAdditionalCover]] = None
    secondarylabel: Optional[str] = None
    placeholder: Optional[str] = None
    radioOptions: Optional[List[IRadioOption]] = None
    selectCheckboxOptions: Optional[List[ISelectCheckboxOption]] = None
    images: Optional[List[IImage]] = None
    otherControlName: Optional[str] = None
    subControls: Optional[List[ISubControl]] = None
    methodName: Optional[str] = None
    getAllOption: Optional[str] = None
    onChangeMethod: Optional[str] = None
    visible: Optional[bool] = True
    conditionalVisibility: Optional[IConditionalVisibility] = None
    popUpFormId: Optional[str] = None
    showDoneButton: Optional[bool] = None
    dependentControls: Optional[List[str]] = None
    toolTipText: Optional[str] = None
    isToolTipVisible: Optional[bool] = None
    imagesrc: Optional[str] = None
    tabs: Optional[List[ITab]] = None
    bigFontValue: Optional[str] = None
    details: Optional[Any] = None
    button: Optional[Any] = None
    icon: Optional[str] = None
    imageUrl: Optional[str] = None
    maxDateLength: Optional[Any] = None
    minDateLength: Optional[Any] = None
    maxLength: Optional[int] = None
    minLength: Optional[int] = None
    inputMaxLength: Optional[int] = None
    visibleToolTip: Optional[bool] = None
    toolTipMessage: Optional[str] = None
    isDefault: Optional[bool] = None
    notMandatory: Optional[bool] = None
    postControlCreationMethod: Optional[str] = None
    dependentAddOnControls: Optional[List[Any]] = None
    visibleCondition: Optional[List[Any]] = None

class ISectionButton(BaseModel):
    label: Optional[str]
    class_: Optional[str]  # Renamed to avoid conflict with Python's `class`
    name: Optional[str]
    visible: bool
    apiEndpoint: Optional[str]
    method: Optional[str]
    controlTypeName: Optional[str]
    formControls: List[IFormControl]
    isVisible: Optional[bool]

class IFormSections(BaseModel):
    sectionTitle: str
    visible: Optional[bool] = True
    apiEndpoint: Optional[str] = None
    controlTypeName: Optional[str] = None
    method: Optional[str] = None
    isVisible: Optional[bool] = True
    formControls: List[IFormControl] = []
    visibleLabel: Optional[bool] = True
    sectionButton: Optional[ISectionButton] = None
    class_: Optional[str] = None
    toolTipText: Optional[str] = None
    urlDependentControls: Optional[Any] = None
    urlPath: Optional[Any] = None
    productFeaturesUrl: Optional[Any] = None

class IForm(BaseModel):
    value: Optional[Any]
    valid: Optional[Any]
    get: Optional[Any]
    formTitle: str
    saveBtnTitle: Optional[str]
    saveBtnFunction: Optional[str]
    resetBtnTitle: Optional[str]
    calculateBtnTitle: Optional[str]
    prevBtnTitle: Optional[str]
    themeFile: str
    formSections: List[IFormSections]
    class_: Optional[str]  # Renamed to avoid conflict with Python's `class`